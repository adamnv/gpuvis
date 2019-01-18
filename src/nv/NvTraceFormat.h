/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <stdint.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>


namespace NvTraceFormat {

static const std::array<char, 8> nvtrc01Magic{"nvtrc01"};
struct FileHeader
{
    std::array<char, 8> magic;
};

struct ArrayHeader
{
    int32_t count;
    int32_t elementSize;
};

enum class GpuCtxSwTraceError8 : uint8_t
{
    None = 0,
    UnsupportedGpu = 1,
    UnsupportedDriver = 2,
    NeedRoot = 3,
    Unknown = 255,
};

struct DeviceDesc
{
    uint8_t uuid[16]; // Corresponds to VkPhysicalDeviceIDProperties::deviceUUID
    char name[239];   // Null-terminated string in fixed-size buffer
    GpuCtxSwTraceError8 gpuCtxSwTraceError;
    int64_t cpuTimestampStart;  // On x86, RDTSC
    int64_t gpuTimestampStart;  // NVIDIA GPU globaltimer
    int64_t cpuTimestampEnd;    // On x86, RDTSC
    int64_t gpuTimestampEnd;    // NVIDIA GPU globaltimer
};

enum class Category16 : uint16_t
{
    Invalid = 0,
    GpuContextSwitch = 1,
};

enum class TypeGpuCtxSw16 : uint16_t
{
    Invalid = 0,
    ContextSwitchedIn = 1,
    ContextSwitchedOut = 2,
};

struct RecordGpuCtxSw
{
    Category16      category;
    TypeGpuCtxSw16  type;
    uint32_t        processId;
    int64_t         timestamp;
    uint64_t        contextHandle;
};

struct FileData
{
    std::vector<DeviceDesc> deviceDescs;
    std::vector<std::vector<RecordGpuCtxSw>> perDeviceData; // First index is device, second is record
};

template <class T>
inline void Read(std::ifstream& ifs, T& value, int sizeOfValue = sizeof(T))
{
    ifs.read(reinterpret_cast<char*>(&value), sizeOfValue);
}

template <class T, class Alloc>
inline bool ReadVector(std::ifstream& ifs, std::vector<T, Alloc>& buffer)
{
    ArrayHeader header;
    Read(ifs, header);
    if (!ifs) return false;

    buffer.resize(header.count);

    if (header.elementSize == sizeof(T))
    {
        ifs.read(reinterpret_cast<char*>(&buffer[0]), header.count * header.elementSize);
    }
    else if (header.elementSize > sizeof(T))
    {
        for (T& elem : buffer)
        {
            ifs.read(reinterpret_cast<char*>(&elem), sizeof(T));
        }
    }
    else
    {
        // File has older version than expected.
        // Could attempt to upconvert, but for now simply fail.
        return false;
    }

    if (!ifs) return false;

    return true;
}

template <class T>
inline void Write(std::ofstream& ofs, T& value)
{
    ofs.write(reinterpret_cast<char*>(&value), sizeof(T));
}

template <class T, class Alloc>
inline void WriteVector(std::ofstream& ofs, std::vector<T, Alloc> const& buffer)
{
    ArrayHeader header{(int32_t)buffer.size(), (int32_t)sizeof(T)};
    Write(ofs, header);

    ofs.write(reinterpret_cast<char const*>(&buffer[0]), header.count * header.elementSize);
}

inline bool ReadFileData(char const* inputFile, FileData& fileData)
{
    fileData = FileData();

    std::ifstream ifs(inputFile, std::ios::in | std::ios::binary);
    if (!ifs) return false;

    FileHeader header;
    Read(ifs, header);
    if (!ifs) return false;

    if (header.magic != nvtrc01Magic) return false;

    bool success = ReadVector(ifs, fileData.deviceDescs);
    if (!success) return false;

    fileData.perDeviceData.resize(fileData.deviceDescs.size());
    for (auto& deviceData : fileData.perDeviceData)
    {
        bool success = ReadVector(ifs, deviceData);
        if (!success) return false;
    }

    return true;
}

inline bool WriteFileData(char const* outputFile, FileData const& fileData)
{
    std::ofstream ofs(outputFile, std::ios::out | std::ios::binary);
    if (!ofs) return false;

    FileHeader header{nvtrc01Magic};
    Write(ofs, header);

    WriteVector(ofs, fileData.deviceDescs);

    for (auto const& deviceData : fileData.perDeviceData)
    {
        WriteVector(ofs, deviceData);
    }

    if (!ofs) return false;

    return true;
}

struct TimestampConverter
{
    int64_t dstAtSyncPoint;
    int64_t srcAtSyncPoint;
    double scale;

    int64_t operator()(int64_t srcTimestamp) const
    {
        auto srcDelta = srcTimestamp - srcAtSyncPoint;
        auto dstDelta = static_cast<int64_t>(scale * static_cast<double>(srcDelta));
        return dstDelta + dstAtSyncPoint;
    }
};

inline TimestampConverter CreateTimestampConverter(
    int64_t srcStart,
    int64_t srcEnd,
    int64_t dstStart,
    int64_t dstEnd)
{
    auto dstDelta = dstEnd - dstStart;
    auto srcDelta = srcEnd - srcStart;
    double scale = (srcDelta == 0.0) ? 0.0 :
        (static_cast<double>(dstDelta) / static_cast<double>(srcDelta));

    // Any sync point can be used for conversions.  Since we are subtracting the
    // sync point from each timestamp before scaling it, the 53-bit mantissa of
    // double makes our scaling precision about 1 nanosecond per week of distance
    // from the sync point (assuming 1 GHz clocks).  So, accuracy is best near
    // the sync point, samples one week later could be off by 1ns, samples two
    // weeks later by 2ns, etc.  For short captures the choice of sync point is
    // irrelevant, but in a snapshot-based tool where the region of interest is
    // nearer to the end, we should prioritize accuracy at the end highest, so we
    // select the end of capture as our sync point.

    return {dstEnd, srcEnd, scale};
}

// Helpful overload for the simple case of creating a converter based on the
// oldest and newest known sync points for a given device.
inline TimestampConverter CreateTimestampConverter(DeviceDesc const& desc)
{
    // Source is GPU time, destination is CPU time
    return CreateTimestampConverter(
        desc.gpuTimestampStart,
        desc.gpuTimestampEnd,
        desc.cpuTimestampStart,
        desc.cpuTimestampEnd);
}

// Note that timestamps are automatically converted to CPU time unless raw GPU
// timestamps were explicitly requested.  The automatic conversion effectively
// works like this:
// for (size_t deviceIndex = 0; deviceIndex < deviceDescs.size(); ++deviceIndex)
// {
//     auto const& deviceDesc = fileData.deviceDescs[deviceIndex];
//     auto& records = fileData.perDeviceData[deviceIndex];
//
//     auto convertToCpuTime = CreateTimestampConverter(deviceDesc);
//
//     for (auto& record : records)
//         record.timestamp = convertToCpuTime(record.timestamp);
// }
//
// In the case of merging multiple FileData objects onto a single timeline, it
// is most accurate to leave all the timestamps in GPU time, and then convert
// them all afterwards using a single conversion factor.  Create this common
// converter using the start time of the earliest capture and the end time of
// the latest capture (remembering to handle this separately for each device).

inline void SetName(DeviceDesc& desc, std::string const& name)
{
    // Avoid min() macro trouble
    auto quickMin = [](size_t a, size_t b) { return a < b ? a : b; };

    // Truncate name if too long, ensuring there's a null terminator
    size_t indexOfNull = quickMin(name.size(), sizeof(desc.name) - 1);
    memcpy(&desc.name[0], name.c_str(), indexOfNull);
    desc.name[indexOfNull] = '\0';
}

// Assume uuid points to 16-byte array
inline std::string PrintableUuid(uint8_t const* uuid)
{
    std::ostringstream oss;
    oss << std::hex;

    int offset = 0;
    auto printBytes = [&](int count)
    {
        int end = offset + count;
        for (int i = offset; i < end; ++i)
        {
            // Cast to avoid potentially printing a char type as ASCII
            oss << static_cast<uint16_t>(uuid[i]);
        }
        offset = end;
    };

    printBytes(4);
    oss << '-';
    printBytes(2);
    oss << '-';
    printBytes(2);
    oss << '-';
    printBytes(2);
    oss << '-';
    printBytes(6);

    return oss.str();
}

inline void PrettyPrintFileData(
    std::ostream& os,
    FileData const& fileData,
    bool showDeviceDescs = true,
    bool showRecords = true)
{
    int d;

    if (showDeviceDescs)
    {
        d = 0;
        for (auto& desc : fileData.deviceDescs)
        {
            const char* ctxswSupportedMsg;
            switch (desc.gpuCtxSwTraceError)
            {
            case GpuCtxSwTraceError8::None:
                ctxswSupportedMsg = "yes";
                break;
            case GpuCtxSwTraceError8::UnsupportedGpu:
                ctxswSupportedMsg = "no -- unsupported GPU (requires Volta, Turing, or newer)";
                break;
            case GpuCtxSwTraceError8::UnsupportedDriver:
                ctxswSupportedMsg = "no -- driver is missing required support, try a newer version";
                break;
            case GpuCtxSwTraceError8::NeedRoot:
                ctxswSupportedMsg = "no -- process must be running as root/admin to use this feature";
                break;
            case GpuCtxSwTraceError8::Unknown: [[fallthrough]];
            default:
                ctxswSupportedMsg = "no -- internal error encountered";
                break;
            }

            os << "Device " << d << ":\n"
                "\tName: " << &desc.name[0] << "\n"
                "\tUUID: {" << PrintableUuid(&desc.uuid[0]) << "}\n"
                "\tSupports GPU context-switch trace: " << ctxswSupportedMsg << "\n"
                "\tTimestamps for synchronization (raw values, in hex):\n" << std::hex << std::right <<
                "\t  CPU start: " << desc.cpuTimestampStart << " GPU start: " << desc.gpuTimestampStart << "\n"
                "\t  CPU end:   " << desc.cpuTimestampEnd   << " GPU end:   " << desc.gpuTimestampEnd   << "\n" <<
                std::dec << std::left;
            ++d;
        }
    }

    if (showRecords)
    {
        d = 0;
        for (auto& records : fileData.perDeviceData)
        {
            os << "Device " << d << " records:\n";
            ++d;
            for (auto& record : records)
            {
                char const* type =
                    (record.type == TypeGpuCtxSw16::ContextSwitchedIn) ? "Context Start" :
                    (record.type == TypeGpuCtxSw16::ContextSwitchedOut) ? "Context Stop" :
                    "<Other>";
                os << "\tTimestamp: 0x" << std::setw(16) << std::setfill('0') << std::right << std::hex << record.timestamp << std::dec <<
                    " | Event: " << std::setw(13) << std::setfill(' ') << std::left << type <<
                    " | PID: " << std::setw(10) << record.processId <<
                    " | ContextID: 0x" << std::setw(8) <<  std::setfill('0') << std::right << std::hex << record.contextHandle << std::dec <<
                    "\n";

            }
        }
    }
}

} // namespace
