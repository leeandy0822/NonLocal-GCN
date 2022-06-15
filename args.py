#!/usr/bin/env python3

import argparse
import torch
import os 

class Option():
    def __init__(self):
        
        parser = argparse.ArgumentParser(prog="DLP homework 7", description='This lab will implement cgan')
        parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate")
        parser.add_argument("--epochs", type=int, default=15, help="Epoch")
        parser.add_argument("--batch", type=int, default=100, help="Batch size")
        parser.add_argument("--NonLocal_Operator", type=str, default="None", help="embed_gaussian, gaussian, dot, concate")
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save folder")
        parser.add_argument("--load_model",  type=str, default= None, help="wandb weight link ")
        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        print(args)
        return args 