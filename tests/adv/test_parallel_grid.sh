mpirun -np 2 python test_parallel_grid.py order_2_4 1
mpirun -np 4 python test_parallel_grid.py order_2_4 1000

mpirun -np 4 python test_parallel_grid.py order_4_2 1
mpirun -np 2 python test_parallel_grid.py order_4_2 1000

mpirun -np 2 python test_parallel_grid.py order_2_2 1
mpirun -np 2 python test_parallel_grid.py order_2_2 1000

mpirun -np 4 python test_parallel_grid.py order_4_4 1
mpirun -np 4 python test_parallel_grid.py order_4_4 1000

mpirun -np 4 python test_parallel_grid.py order_4 1000



