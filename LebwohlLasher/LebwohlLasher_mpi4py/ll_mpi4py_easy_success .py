from mpi4py import MPI
import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# =======================================================================
def initdat(nmax):
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    return arr


def plotdat(arr, pflag, nmax):
    if rank == 0 and pflag != 0:
        u = np.cos(arr)
        v = np.sin(arr)
        x = np.arange(nmax)
        y = np.arange(nmax)
        cols = np.zeros((nmax, nmax))
        # ... (rest of the plotting code, but modified to run only on rank 0)


def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"LL-Output-{current_datetime}-Rank{rank}.txt"
    # ... (adapted to save data with rank identifier)


def one_energy(arr, ix, iy, nmax):
    en = 0.0
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax
    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    # ... (compute energy contributions from all neighbors)
    return en


def all_energy(arr, nmax):
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall


def get_order(arr, nmax):
    Qab = np.zeros((3, 3))
    delta = np.eye(3, 3)
    lab = np.vstack((np.cos(arr), np.sin(arr), np.zeros_like(arr))).reshape(3, nmax, nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a, b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
    Qab = Qab / (2 * nmax * nmax)
    eigenvalues, _ = np.linalg.eig(Qab)
    return np.max(eigenvalues)


def MC_step(arr, Ts, nmax):
    scale = 0.1 + Ts
    accept = 0
    xran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    yran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    aran = np.random.normal(scale=scale, size=(nmax, nmax))
    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            if en1 <= en0:
                accept += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                if boltz >= np.random.uniform(0.0, 1.0):
                    accept += 1
                else:
                    arr[ix, iy] -= ang
    return accept / (nmax * nmax)


def run_simulation(nsteps, nmax, temp):
    lattice = initdat(nmax)
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    initial = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    final = time.time()
    runtime = final - initial

    return lattice, ratio, energy, order, runtime


def main (program, nsteps, nmax, temp, pflag):
    lattice, ratio, energy, order, runtime = run_simulation(ITERATIONS, SIZE, TEMPERATURE)
    if rank == 0:
        lattice_list = comm.gather(lattice, root=0)
        ratio_list = comm.gather(ratio, root=0)
        energy_list = comm.gather(energy, root=0)
        order_list = comm.gather(order, root=0)
        runtime_list = comm.gather(runtime, root=0)

        orders = np.array([o[-1] for o in order_list])
        ratios = np.array([r[-1] for r in ratio_list])
        avg_order = np.mean(orders)
        std_order = np.std(orders)
        avg_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
        filename_prefix = f"LL-MPI-Output-{current_datetime}"
        for i, (lat, r, e, o, t) in enumerate(zip(lattice_list, ratio_list, energy_list, order_list, runtime_list)):
            savedat(lat, ITERATIONS, TEMPERATURE, t, r, e, o, SIZE, i)

        if PLOTFLAG != 0:
            plotdat(lattice_list[0], PLOTFLAG, SIZE)


if __name__ == '__main__':
    start_time=time.time()
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time)} ")