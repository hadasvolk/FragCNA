import os
import pickle
import configparser

from read_counter import ReadCounter

class PanelOfNormals:
    def __init__(self, config):
        self.bam_files = config.get('PON', 'bam_files').split(',')
        self.pon_path = config['PATH']['pon_path']
        os.makedirs(self.pon_path, exist_ok=True)
        self.read_counter_config = config

    def process_bam_files(self):
        for bam_file in self.bam_files:
            bam_file_name = os.path.basename(bam_file)
            bam_file_name_no_ext = os.path.splitext(bam_file_name)[0]
            print(f'Processing BAM file: {bam_file}')
            self.read_counter_config.set('PATH', 'bam_file', bam_file)
            self.read_counter_config.set('PARAMS', 'name', bam_file_name_no_ext)
            read_counter = ReadCounter(config=self.read_counter_config)
            regions, all_fragment_length_freq = read_counter.rd()
            with open(os.path.join(self.pon_path, f'{bam_file_name_no_ext}.regions.pkl'), 'wb') as f:
                pickle.dump(regions, f)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    panel_of_normals = PanelOfNormals(config=config)
    panel_of_normals.process_bam_files()

