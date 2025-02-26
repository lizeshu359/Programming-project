from mpi4py import MPI
import sys
import time
import datetime
import numpy as np

from LebwohlLasher_numba.run import start_time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# =======================================================================
def initdat(nmax, local_nmax, rank, comm):
    """
    Initialize the lattice with random angles and ghost layers using MPI.
    """
    if rank == 0:
        full_arr = np.random.random((nmax, nmax)) * 2.0 * np.pi
    else:
        full_arr = None

    # Scatter data to individual processes
    sendcounts = [nmax * local_nmax] * size
    displacements = [i * nmax * local_nmax for i in range(size)]
    local_arr = np.empty((local_nmax + 2, nmax), dtype=np.float64)

    comm.Scatterv([full_arr, sendcounts, displacements, MPI.DOUBLE], local_arr[1:-1, :], root=0)

    # Initialize ghost layers (periodic boundaries)
    if rank == 0:
        # Bottom ghost layer (receive from bottom neighbor)
        if size > 1:
            comm.Recv(local_arr[-1, :], source=rank + 1, tag=12)
        # Top ghost layer (copy from local_arr[1])
        local_arr[0, :] = local_arr[1, :]
    elif rank == size - 1:
        # Top ghost layer (receive from top neighbor)
        if size > 1:
            comm.Recv(local_arr[0, :], source=rank - 1, tag=11)
        # Bottom ghost layer (copy from local_arr[-2])
        local_arr[-1, :] = local_arr[-2, :]
    else:
        # Bottom ghost layer (receive from bottom neighbor)
        comm.Recv(local_arr[-1, :], source=rank + 1, tag=12)
        # Top ghost layer (receive from top neighbor)
        comm.Recv(local_arr[0, :], source=rank - 1, tag=11)

    return local_arr


# =======================================================================
def update_ghost_layers(local_arr, comm):
    """
    Update ghost layers with periodic boundary conditions using MPI.
    """
    if rank == 0:
        # Bottom neighbor (rank 1) sends top data
        if size > 1:
            comm.Send(local_arr[1, :], dest=rank + 1, tag=11)
        # Top ghost layer (local_arr[0]) is copied from local_arr[1]
        local_arr[0, :] = local_arr[1, :]
    elif rank == size - 1:
        # Top neighbor (rank - 1) sends bottom data
        if size > 1:
            comm.Send(local_arr[-2, :], dest=rank - 1, tag=12)
        # Bottom ghost layer (local_arr[-1]) is copied from local_arr[-2]
        local_arr[-1, :] = local_arr[-2, :]
    else:
        # Send top data to bottom neighbor
        comm.Send(local_arr[1, :], dest=rank - 1, tag=12)
        # Send bottom data to top neighbor
        comm.Send(local_arr[-2, :], dest=rank + 1, tag=11)
        # Receive new top ghost layer from top neighbor
        comm.Recv(local_arr[0, :], source=rank - 1, tag=11)
        # Receive new bottom ghost layer from bottom neighbor
        comm.Recv(local_arr[-1, :], source=rank + 1, tag=12)


# =======================================================================
def one_energy(arr, ix, iy, nmax):
    """
    Calculate the energy of a single cell with periodic boundaries.
    """
    en = 0.0
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    return en


# =======================================================================
def all_energy(arr, nmax, local_nmax):
    """
    Calculate the total energy of the local lattice region.
    """
    enall = 0.0
    start_row = 1
    end_row = local_nmax + 1
    for i in range(start_row, end_row):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall


# =======================================================================
def get_order(arr, nmax, local_nmax):
    """
    Calculate the order parameter for the local lattice region.
    """
    Qab = np.zeros((3, 3))
    delta = np.eye(3, 3)
    lab = np.vstack((np.cos(arr), np.sin(arr), np.zeros_like(arr))).reshape(3, nmax, nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a, b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
    Qab = Qab / (2 * nmax * nmax)
    eigenvalues = np.linalg.eigvals(Qab)
    return np.max(eigenvalues)


# =======================================================================
def MC_step(local_arr, Ts, nmax, local_nmax, comm):
    """
    Perform one Monte Carlo step on the local lattice region.
    """
    scale = 0.1 + Ts
    accept = 0

    xran = np.random.randint(0, high=local_nmax, size=local_nmax * nmax)
    yran = np.random.randint(0, high=nmax, size=local_nmax * nmax)
    aran = np.random.normal(scale=scale, size=local_nmax * nmax)

    for a in range(local_nmax * nmax):
        ix_local = xran[a] + 1  # Adjust for local grid (without ghost layers)
        iy = yran[a]
        ang = aran[a]

        # Calculate energy before change
        en0 = one_energy(local_arr, ix_local, iy, nmax)

        # Propose change
        local_arr[ix_local, iy] += ang

        # Calculate energy after change
        en1 = one_energy(local_arr, ix_local, iy, nmax)

        if en1 <= en0 or np.exp(-(en1 - en0) / Ts) >= np.random.uniform(0.0, 1.0):
            accept += 1
        else:
            # Reject change
            local_arr[ix_local, iy] -= ang

    # Update ghost layers after MC step
    update_ghost_layers(local_arr, comm)

    return accept / (local_nmax * nmax)


# =======================================================================
def main(program, nsteps, nmax, temp, pflag, comm):
    """
    Main simulation function using MPI parallelization.
    """
    if rank == 0:
        print("Starting simulation...")

    local_nmax = nmax // size
    if local_nmax == 0:
        print("Error: Not enough processes for lattice size", nmax)
        comm.Abort()

    lattice = initdat(nmax, local_nmax, rank, comm)

    # Energy and order arrays
    energy = np.zeros(nsteps + 1)
    order = np.zeros(nsteps + 1)
    ratio = np.zeros(nsteps + 1)

    # Initial energy and order
    energy[0] = all_energy(lattice, nmax, local_nmax)
    order[0] = get_order(lattice, nmax, local_nmax)
    ratio[0] = 0.5  # Initial acceptance ratio estimate

    # Perform MC steps
    initial = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax, local_nmax, comm)
        energy[it] = all_energy(lattice, nmax, local_nmax)
        order[it] = get_order(lattice, nmax, local_nmax)

    final = time.time()
    runtime = final - initial

    # Gather results from all processes
    energy = comm.gather(energy, root=0)
    order = comm.gather(order, root=0)
    ratio = comm.gather(ratio, root=0)

    if rank == 0:
        # Calculate global averages
        merged_energy = np.mean(energy, axis=0)
        merged_order = np.mean(order, axis=0)
        merged_ratio = np.mean(ratio, axis=0)

        print(f"Simulation complete. Runtime: {runtime:.3f} seconds.")
        print(f"Final Order Parameter: {merged_order[-1]:.4f}")


# =======================================================================
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: mpirun -n <cores> python script.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
        sys.exit(1)

    PROGNAME = sys.argv[0]
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])
    start_time=time.time()
    main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, comm)
    end_time=time.time()
    print(f'{start_time-end_time}')