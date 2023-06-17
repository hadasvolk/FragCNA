import os
import pickle
import configparser
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from read_counter import ReadCounter

class PanelOfNormals:
    def __init__(self, config):
        self.processes = config.getint('DEFAULT', 'processes')
        self.bam_files = config.get('PON', 'bam_files').split(',')
        self.pon_path = config['PATH']['pon_path']
        os.makedirs(self.pon_path, exist_ok=True)
        self.read_counter_config = config


    def sample_to_process(self, bam_file, t='BAM'):
        bam_file_name = os.path.basename(bam_file)
        bam_file_name_no_ext = os.path.splitext(bam_file_name)[0]
        print(f'Processing {t} file: {bam_file_name_no_ext}')
        self.read_counter_config.set('PATH', 'bam_file', bam_file)
        self.read_counter_config.set('PARAMS', 'name', bam_file_name_no_ext)
        self.read_counter_config.set('PATH', 'outpath', self.pon_path)
        return ReadCounter(config=self.read_counter_config), bam_file_name_no_ext


    def process_bam_files(self):
        for bam_file in self.bam_files:
            read_counter, bam_file_name_no_ext = self.sample_to_process(bam_file, t='Pickle')
            regions, all_fragment_length_freq = read_counter.rd()


    def process_pkl_file(self, bam_file):
        read_counter, bam_file_name_no_ext = self.sample_to_process(bam_file)
        read_counter.read_rd()
        regions = read_counter.process_regions()
        regions = regions[['chrom', 'start', 'end', 'log2']]
        regions.rename(columns={'log2': f'{bam_file_name_no_ext}_log2'}, inplace=True)
        return regions
    

    def read_rd(self):
        with ProcessPoolExecutor(max_workers=self.processes) as executor:
            median_coverages = list(executor.map(self.process_pkl_file, self.bam_files))
        
        median_coverage_df = pd.concat(median_coverages, axis=1)
        median_coverage_df['median'] = median_coverage_df.median(axis=1, skipna=True)
        with open(os.path.join(self.pon_path, 'median_coverage.pkl'), 'wb') as f:
            pickle.dump(median_coverage_df, f)
        return median_coverage_df


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    panel_of_normals = PanelOfNormals(config=config)
    median = panel_of_normals.read_rd()
    print(median)

