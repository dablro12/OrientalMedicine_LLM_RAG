import os 
import sys 

sys.path.append('../')
import streamlit as st
from utils.embedding import binary_embed_file, multi_embed_files