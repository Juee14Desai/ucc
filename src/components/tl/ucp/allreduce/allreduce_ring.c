/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_coll_utils.h"
#include "../reduce_scatter/reduce_scatter.h"
#include "../allgather/allgather.h"

/* Allreduce Ring = Reduce-Scatter Ring + Allgather Ring
 *
 * Phase 1 (reduce_scatter_ring): each rank reduces all data and ends up
 *         with its own block at offset rrank * block_count in dst.
 * Phase 2 (allgather_ring): in-place allgather collects all blocks into dst.
 *
 * If the allreduce is not in-place, src is copied to dst before phase 1
 * so that reduce_scatter can run in-place on dst.
 */

static ucc_status_t
ucc_tl_ucp_allreduce_ring_frag_start(ucc_coll_task_t *task)
{
    ucc_coll_args_t *args = &task->bargs.args;

    if (!UCC_IS_INPLACE(*args)) {
        size_t       dt_size = ucc_dt_size(args->dst.info.datatype);
        size_t       count   = args->dst.info.count;
        ucc_status_t status;

        status = ucc_mc_memcpy(args->dst.info.buffer, args->src.info.buffer,
                               count * dt_size, args->dst.info.mem_type,
                               args->src.info.mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    return ucc_schedule_start(task);
}

static ucc_status_t
ucc_tl_ucp_allreduce_ring_frag_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_ring_init(ucc_base_coll_args_t *coll_args,
                               ucc_base_team_t      *team,
                               ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t    *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t  rs_args  = *coll_args;
    ucc_base_coll_args_t  ag_args  = *coll_args;
    ucc_schedule_t       *schedule;
    ucc_coll_task_t      *rs_task, *ag_task;
    ucc_status_t          status;

    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                     (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    /* Phase 1: reduce_scatter_ring
     * We set up as in-place; the start function does memcpy if needed.
     * Both src and dst counts = allreduce count so rs_ring_total_count works. */
    rs_args.args.coll_type = UCC_COLL_TYPE_REDUCE_SCATTER;
    rs_args.args.src.info.buffer = coll_args->args.dst.info.buffer;
    rs_args.args.src.info.count  = coll_args->args.dst.info.count;
    rs_args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
    rs_args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    UCC_CHECK_GOTO(ucc_tl_ucp_reduce_scatter_ring_init(&rs_args, team,
                                                        &rs_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, rs_task), out, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, rs_task,
                                          UCC_EVENT_SCHEDULE_STARTED),
                   out, status);

    /* Phase 2: allgather_ring (in-place on dst buffer) */
    ag_args.args.coll_type = UCC_COLL_TYPE_ALLGATHER;
    ag_args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
    ag_args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    UCC_CHECK_GOTO(ucc_tl_ucp_allgather_ring_init(&ag_args, team, &ag_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, ag_task), out, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(rs_task, ag_task,
                                          UCC_EVENT_COMPLETED),
                   out, status);

    schedule->super.post     = ucc_tl_ucp_allreduce_ring_frag_start;
    schedule->super.finalize = ucc_tl_ucp_allreduce_ring_frag_finalize;
    *task_h = &schedule->super;
    return UCC_OK;
out:
    return status;
}
