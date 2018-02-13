#!/usr/bin/env python3

import os

tech_stocks = ["fb", "appl", "googl", "intc", "amd", "nvda", "kodk", "mu", "baba", "amzn", "hpq", "dvmt" ]

#https://investingnews.com/daily/tech-investing/blockchain-investing/blockchain-technology-stocks/
blockchain_stocks = ["code", "btcs", "btl", "coin", "dcc", "xblk", "bloc", "kash", "hive", "mara", "mgti", "dash"]

source_filepath = "../Data/Stocks/"
output_filepath = "../ExtractedData/Stocks/"

for root, dirs, files in os.walk(source_filepath):
    for filename in files:
        ticker = filename.split(".")[0]
        if ticker in tech_stocks:
            input_file = open(source_filepath + filename, 'r')
            output_file = None
            usable = False
            for line in input_file:
                year = line[:4]
                if usable:
                    outputFile.write(line)
                elif year >= "2008" and year.isnumeric():
                    print("GOOD: " + line)
                    usable = True
                    outputFile = open(output_filepath + filename, 'w+')
                    outputFile.write(line)
            if output_file != None:
                output_file.close()
