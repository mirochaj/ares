mpirun -np 4 python test_mc_repeat.py test_mc_1 1234
mpirun -np 4 python test_mc_repeat.py test_mc_2 1234
mpirun -np 4 python test_mc_repeat.py test_mc_3 2468
python test_mc_repeat_results.py
