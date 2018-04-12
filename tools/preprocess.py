# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:48:11 2018

@author: Cyrus Dsouza
"""

import re
import numpy as np
strip_special_characters = re.compile("[^A-Za-z0-9 ]+")

ROOT = "D:/Cyrus/Implementations//Sentiment-Analysis-with-LSTM"
        

def clean(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_characters, "", string.lower())


if __name__ == "__main__":
    clean("Yesterday there were many ! such people in the @ house")

