"""Exercise the shim's marker loop with discarded and delayed recordings."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InputReadinessTests(unittest.TestCase):
    def test_exact_input_slot_before_evaluation(self):
        source = (ROOT / 'tools/d4r_nvngx_shim.cpp').read_text()
        start = source.index('    // A later input marker')
        end = source.index('    const auto ready = ProfileClock::now();', start)
        runner_source = r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <thread>
using ProfileClock = std::chrono::steady_clock;
struct Feature { uint32_t* markerValue; std::atomic<bool> retiring{false}; };
struct InputSlot { bool busy = true; };
static unsigned retired = 0, evaluated = 0, sleeps = 0;
static uint32_t* arrival = nullptr;
static uint32_t arrivalFrame = 0;
static void frame_retired(Feature*, uint32_t) { ++retired; }
static void logf(const char*, ...) {}
static unsigned env_uint(const char*, unsigned fallback) { return fallback; }
static void d4r_sleep_us(unsigned) {
    assert(++sleeps <= 2);
    assert(arrival != nullptr);
    *arrival = arrivalFrame;
}
static void prepare(Feature* feature, int slotIndex, uint32_t frame, InputSlot& slot) {
    const auto start = ProfileClock::now();
''' + source[start:end] + r'''
    ++evaluated;
}
int main() {
    // A later submission advances the aggregate marker, but not frame 1's
    // slot. Old aggregate-only logic would evaluate unsubmitted inputs here.
    uint32_t markers[] = {2, 0, 0, 2};
    Feature feature{markers};
    InputSlot skipped;
    prepare(&feature, 1, 1, skipped);
    assert(retired == 1 && evaluated == 0 && sleeps == 0 && !skipped.busy);
    InputSlot ready;
    prepare(&feature, 2, 2, ready);
    assert(retired == 1 && evaluated == 1 && sleeps == 0);
    // Delayed input stays pending until that slot's exact tag arrives.
    markers[0] = 3;
    arrival = &markers[1];
    arrivalFrame = 3;
    InputSlot delayed;
    prepare(&feature, 0, 3, delayed);
    assert(evaluated == 2 && sleeps == 1);
    feature.retiring = true;
    InputSlot cancelled;
    prepare(&feature, 1, 4, cancelled);
    assert(retired == 2 && evaluated == 2 && !cancelled.busy);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / 'input-readiness'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-x', 'c++', '-', '-o', str(runner)], input=runner_source,
                           text=True, capture_output=True, check=True)
            subprocess.run([str(runner)], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
