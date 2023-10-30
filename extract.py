from itertools import zip_longest
import re


with open('out1', 'r') as file:
    epoch = 1
    printf(f"Epoch\tLoss\tTrain_accuracy\tTest_accuracy")
    for line1, line2 in zip_longest(*[file] * 2, fillvalue=''):
        line1 = line1.strip()
        line2 = line2.strip()

        lossmatch = re.search(r'\d+\.\d+', line1)
        if lossmatch:
            loss = lossmatch.group()

        accmatch = re.search(r'\d+\.\d+\%', line1)
        if accmatch:
            acc = accmatch.group()

        testaccmatch = re.search(r'\d+\.\d+\%', line2)
        if testaccmatch:
            testacc = testaccmatch.group()

        print(f"{epoch}\t{loss}\t{acc}\t{testacc}")
        epoch += 1


