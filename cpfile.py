import shutil

train_dir="trainset"
test_dir="testset"

with open("trainlist", "r") as file:
	for line in file:
		line = line.strip()
		shutil.copy(line, train_dir)

with open("testlist", "r") as file:
	for line in file:
		line = line.strip()
		shutil.copy(line, test_dir)




