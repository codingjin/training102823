trainlabelfile = "luad_train_label"
testlabelfile = "luad_test_label"

train_list_file = "trainlist"
test_list_file = "testlist"

f_train = open(trainlabelfile, "w")
f_test = open(testlabelfile, "w")

with open(train_list_file, "r") as file:
	for line in file:
		line = line.strip()
		f_train.write(f"{line}\t0\n")
	
f_train.close()

with open(test_list_file, "r") as file:
	for line in file:
		line = line.strip()
		f_test.write(f"{line}\t0\n")

f_test.close()



