/*
 * Copyright 2018 Valve Software
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// almost all of these includes exist solely to support trace-read.h so we can use trace_info_t; maybe we should just avoid trace_info_t

#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>

#include <cstring>
#include <vector>
#include <forward_list>
#include <csetjmp>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <future>

#include <sys/mman.h>
#include <sys/param.h>
#include <unistd.h>

#ifdef __APPLE__
#define lseek64 lseek
#define off64_t off_t
#endif

#include "gpuvis_macros.h"

#include "trace-cmd/trace-read.h" // for trace_info_t and EventCallback

#include "nv/NvTraceFormat.h"

int64_t rdtsc_to_us(int64_t rdtsc)
{
    // checkme: might not be right, is rdtsc accuracy defined?
    return int64_t(rdtsc);
}

void adapt_trace_info(trace_info_t &trace_info_dest, NvTraceFormat::FileData &fileData_src, const char *filename)
{
    int64_t min_filedata_cpu_rdtsc = INT64_MAX;

    for ( size_t i=0; i<fileData_src.deviceDescs.size(); ++i )
    {
        if ( i>0 )
        {
            trace_info_dest.uname += "&";
        }
        trace_info_dest.uname += "nvgpu(" + std::string(fileData_src.deviceDescs[i].name) + ")";

        min_filedata_cpu_rdtsc = std::min(min_filedata_cpu_rdtsc, fileData_src.deviceDescs[i].cpuTimestampStart);
    }

    trace_info_dest.timestamp_in_us = true;
    trace_info_dest.cpus = 0;
    trace_info_dest.file = filename;
    trace_info_dest.min_file_ts = rdtsc_to_us(min_filedata_cpu_rdtsc); // perhaps just don't bother, check where this is used
}

void adapt_events(EventCallback &cb, trace_info_t &trace_info, NvTraceFormat::FileData &fileData_src, StrPool &strpool)
{
    for ( int i=0; i<fileData_src.deviceDescs.size(); ++i)
    {
        for ( NvTraceFormat::RecordGpuCtxSw &record : fileData_src.perDeviceData[i] )
        {
            const std::string namehack[] = {"Invalid", "ContextSwitchedIn", "ContextSwitchedOut"};
            trace_event_t adapted;

            adapted.pid = record.processId;
            adapted.id = INVALID_ID;
            adapted.cpu = 0;
            adapted.ts = rdtsc_to_us(record.timestamp);

            adapted.flags = TRACE_FLAG_AUTOGEN_COLOR; // maybe TRACE_FLAG_SCHED_SWITCH, swqueue, hwqueue...
            adapted.seqno = 0; //?
            adapted.id_start = INVALID_ID;
            adapted.graph_row_id = 0;
            adapted.crtc = -1;

            adapted.color = 0; // == default

            adapted.duration = INT64_MAX; // == 'not set'

            adapted.comm = strpool.getstr("(event_comm)"); // command name
            adapted.system = strpool.getstr("nvcontext"); // event system
            adapted.name = strpool.getstr(std::string("(event_name:" + namehack[int(record.type)] + ")").c_str()); // event name
            adapted.user_comm = strpool.getstr("(event_usercomm)");

            cb(adapted);
        }
    }
    //assert(false); // todo
}

bool read_nvtrc_file(const char *filename, StrPool &strpool, trace_info_t &trace_info, EventCallback &cb)
{
    NvTraceFormat::FileData fileData;

    bool read_success = NvTraceFormat::ReadFileData( filename, fileData );
    if ( read_success )
    {
        for ( size_t i=0; i<fileData.deviceDescs.size(); ++i )
        {
            fprintf( stderr, "nvtrc: %s: GPU device #%d is %s\n", filename, int(i+1), fileData.deviceDescs[i].name );
            fprintf( stderr, "nvtrc: %s: GPU device #%d has %d records\n", filename, int(i+1), int(fileData.perDeviceData[i].size()) );
        }

        adapt_trace_info(trace_info, fileData, filename);
        adapt_events(cb, trace_info, fileData, strpool);
    }

    return read_success;
}
