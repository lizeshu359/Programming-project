import cProfile
import time
import LebwohlLasher_numba

start_time = time.time()
cProfile.run("LebwohlLasher_numba.main(5,100,100, 0.5, 1)")
end_time = time.time()
print(f"Total script runtime: {end_time - start_time:.4f} seconds")