#!/bin/bash

for epoch in {89..138..1}
do
    python test.py -r saved/models/Kimchi-resnet34-aug-346-lr30/1113_115640/checkpoint-epoch${epoch}.pth
done
