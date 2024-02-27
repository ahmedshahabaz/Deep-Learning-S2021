
# 'args.py','dataset.py','models.py' has to be in the same directory as 'train.py' & 'test.py'

# data folder should have the "train.npy" and "test.npy" files

# to train the model run the python script named 'train.py' by using the following command

	python3 train.py

'Robot_Trials_DL.npy' has to be in the "data" directory.


# to test the model run the python script named 'test.py' in the following format
	
	python3 test.py 'test_data.npy'
	
where 'test_data.npy' is the file that contains the test data.

'test_data.npy' has to be in the directory named "data" --> "./data/test_data.npy".

after running the script "test.py" it will create a output numpy file called "shahabaz_out.npy".

ignore other two ".npy" files namely "out_0.npy" & "out_1.npy". These are created for the two different snapshots of the model.

