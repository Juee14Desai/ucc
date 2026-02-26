/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "reduce_scatter.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_dt_reduce.h"
#include "utils/ucc_atomic.h"
#include "coll_patterns/ring.h"

static inline size_t
rs_ring_total_count(ucc_coll_args_t *args)
{
    return UCC_IS_INPLACE(*args) ? args->dst.info.count
                                : args->src.info.count;
}

static inline void send_completion_common(void *request, ucs_status_t status,
                                          void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in rs ring completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ucc_atomic_add32(&task->tagged.send_completed, 1);
    if (request) {
        ucp_request_free(request);
    }
}

static void send_completion_1(void *request, ucs_status_t status,
                              void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->reduce_scatter_ring.s_scratch_busy[0] = 0;
    send_completion_common(request, status, user_data);
}

static void send_completion_2(void *request, ucs_status_t status,
                              void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->reduce_scatter_ring.s_scratch_busy[1] = 0;
    send_completion_common(request, status, user_data);
}

void ucc_tl_ucp_reduce_scatter_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t      *task     = ucc_derived_of(coll_task,
                                                      ucc_tl_ucp_task_t);
    ucc_coll_args_t        *args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t      *team     = TASK_TEAM(task);
    ucc_ring_pattern_t     *ring     = team->cuda_ring;
    ucc_rank_t              rrank    = ucc_ring_pattern_rank(ring, 0);
    ucc_rank_t              tsize    = ucc_ring_pattern_size(ring, 0);
    ucc_rank_t              sendto   = ucc_ring_pattern_get_send_peer(ring, 0,
                                                                      rrank);
    ucc_rank_t              recvfrom = ucc_ring_pattern_get_recv_peer(ring, 0,
                                                                      rrank);
    size_t                  total_count = rs_ring_total_count(args);
    void                   *sbuf     = UCC_IS_INPLACE(*args)
                                           ? args->dst.info.buffer
                                           : args->src.info.buffer;
    ucc_memory_type_t       mem_type = args->dst.info.mem_type;
    ucc_datatype_t          dt       = args->dst.info.datatype;
    size_t                  dt_size  = ucc_dt_size(dt);
    size_t                  block_count = total_count / tsize;
    ucp_send_nbx_callback_t cb[2]    = {send_completion_1, send_completion_2};
    ucc_rank_t              prevblock;
    ucc_status_t            status;
    size_t                  max_block_size, data_displ;
    int                     step, is_avg, id;
    void                   *r_scratch, *s_scratch[2], *reduce_target;
    volatile char          *busy;

    max_block_size = task->reduce_scatter_ring.max_block_count * dt_size;
    busy           = task->reduce_scatter_ring.s_scratch_busy;
    r_scratch      = task->reduce_scatter_ring.scratch;
    s_scratch[0]   = PTR_OFFSET(r_scratch, max_block_size);
    s_scratch[1]   = PTR_OFFSET(s_scratch[0], max_block_size);

    if (UCC_INPROGRESS == ucc_tl_ucp_test_ring(task)) {
        return;
    }
    while (task->tagged.recv_posted > 0) {
        ucc_assert(!busy[0] || !busy[1]);
        id            = busy[0] ? 1 : 0;
        reduce_target = s_scratch[id];
        step          = task->tagged.send_posted;

        prevblock = ucc_ring_pattern_get_send_block(ring, 0, rrank,
                                                    (ucc_rank_t)step);
        ucc_assert(task->tagged.recv_posted == task->tagged.recv_completed);
        ucc_assert(task->tagged.recv_posted < tsize);

        data_displ = prevblock * block_count * dt_size;
        if (task->tagged.recv_completed == tsize - 1) {
            if (UCC_IS_INPLACE(*args)) {
                reduce_target = PTR_OFFSET(args->dst.info.buffer,
                                           rrank * block_count * dt_size);
            } else {
                reduce_target = args->dst.info.buffer;
            }
        }
        is_avg = (args->op == UCC_OP_AVG) &&
                 (task->tagged.recv_completed == (tsize - 1));
        if (UCC_OK !=
            (status = ucc_dt_reduce(
                 r_scratch,
                 PTR_OFFSET(sbuf, data_displ),
                 reduce_target, block_count, dt, args,
                 is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
                 AVG_ALPHA(task), task->reduce_scatter_ring.executor,
                 &task->reduce_scatter_ring.etask))) {
            tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
            task->super.status = status;
            return;
        }
        EXEC_TASK_WAIT(task->reduce_scatter_ring.etask);
        if (task->tagged.recv_completed == tsize - 1) {
            task->tagged.recv_posted = task->tagged.recv_completed = 0;
            break;
        }
        ucc_assert(task->tagged.send_posted - task->tagged.send_completed <= 1);
        ucc_assert(task->tagged.send_posted < tsize);

        busy[id] = 1;
        UCPCHECK_GOTO(ucc_tl_ucp_send_cb(reduce_target,
                                         block_count * dt_size,
                                         mem_type, sendto, team, task,
                                         cb[id], (void *)task),
                      task, out);

        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(r_scratch,
                                         block_count * dt_size,
                                         mem_type, recvfrom, team, task),
                      task, out);

        if (UCC_INPROGRESS == ucc_tl_ucp_test_ring(task)) {
            return;
        }
    }
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }
    task->super.status = UCC_OK;
out:
    return;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t  *team     = TASK_TEAM(task);
    ucc_ring_pattern_t *ring     = team->cuda_ring;
    ucc_rank_t          rrank    = ucc_ring_pattern_rank(ring, 0);
    ucc_rank_t          tsize    = ucc_ring_pattern_size(ring, 0);
    size_t              total_count = rs_ring_total_count(args);
    ucc_datatype_t      dt       = args->dst.info.datatype;
    size_t              dt_size  = ucc_dt_size(dt);
    ucc_memory_type_t   mem_type = args->dst.info.mem_type;
    void               *sbuf     = UCC_IS_INPLACE(*args)
                                       ? args->dst.info.buffer
                                       : args->src.info.buffer;
    size_t              block_count = total_count / tsize;
    ucc_rank_t          sendto   = ucc_ring_pattern_get_send_peer(ring, 0,
                                                                   rrank);
    ucc_rank_t          recvfrom = ucc_ring_pattern_get_recv_peer(ring, 0,
                                                                   rrank);
    ucc_rank_t          send_block;
    size_t              data_displ;
    void               *r_scratch;
    ucc_status_t        status;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    status = ucc_coll_task_get_executor(&task->super,
                                        &task->reduce_scatter_ring.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    r_scratch  = task->reduce_scatter_ring.scratch;

    UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(r_scratch, block_count * dt_size,
                                     mem_type, recvfrom, team, task),
                  task, out);

    send_block = ucc_ring_pattern_get_send_block(ring, 0, rrank, 0);
    data_displ = send_block * block_count * dt_size;
    UCPCHECK_GOTO(ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf, data_displ),
                                     block_count * dt_size, mem_type, sendto,
                                     team, task),
                  task, out);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return task->super.status;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    ucc_mc_free(task->reduce_scatter_ring.scratch_mc_header);
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_reduce_scatter_ring_init_common(
    ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucc_coll_args_t        *args = &TASK_ARGS(task);
    ucc_rank_t              tsize;
    size_t                  total_count, block_count;
    ucc_datatype_t          dt;
    size_t                  dt_size;
    ucc_memory_type_t       mem_type;
    ucc_mc_buffer_header_t *scratch_mc_header;
    ucc_status_t            status;

    if (!team->cuda_ring) {
        return UCC_ERR_NOT_FOUND;
    }

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op &&
        args->op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    tsize       = ucc_ring_pattern_size(team->cuda_ring, 0);
    total_count = rs_ring_total_count(args);
    dt          = args->dst.info.datatype;
    dt_size     = ucc_dt_size(dt);
    mem_type    = args->dst.info.mem_type;
    block_count = total_count / tsize;

    /* 3 blocks: r_scratch (recv), s_scratch[0], s_scratch[1] (double-buffered send) */
    status = ucc_mc_alloc(&scratch_mc_header,
                          block_count * 3 * dt_size, mem_type);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to allocate scratch for reduce_scatter ring");
        return status;
    }

    task->reduce_scatter_ring.scratch           = scratch_mc_header->addr;
    task->reduce_scatter_ring.scratch_mc_header = scratch_mc_header;
    task->reduce_scatter_ring.max_block_count   = block_count;
    task->reduce_scatter_ring.s_scratch_busy[0] = 0;
    task->reduce_scatter_ring.s_scratch_busy[1] = 0;

    task->super.flags   |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_reduce_scatter_ring_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_ring_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_ring_finalize;

    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t      *team,
                                    ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_reduce_scatter_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
