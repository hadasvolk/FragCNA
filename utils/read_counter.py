import os
import sys
import configparser
from multiprocessing import Pool, cpu_count
import pickle

import pandas as pd
import numpy as np
import pysam
import statsmodels.api as sm


def per_region(
        bam_file, 
        fastafile, 
        n, 
        index: int, 
        chrom: str, 
        start: int, 
        end: int, 
        mapq_threshold: int = 20, 
        max_template_length: int = 1000
        ):
    coverage = 0
    sequence_list = []
    fragment_length_list = []
    for read in bam_file.fetch(chrom, start, end):
        if read.mapping_quality >= mapq_threshold:
            if abs(read.template_length) <= max_template_length and read.is_read1 and read.is_proper_pair:
                frag_start = min(read.reference_start, read.next_reference_start)
                frag_end = frag_start + abs(read.template_length)
                template_sequence = fastafile.fetch(chrom, frag_start, frag_end)
                sequence_list.append(template_sequence)
                fragment_length_list.append(int(abs(read.template_length)))
            coverage += 1
    if fragment_length_list:
        fragment_length_freq = np.bincount(np.array(fragment_length_list))
        fragment_length_freq.resize(1000)
        fragment_length_freq = fragment_length_freq / len(fragment_length_list)
    else:
        fragment_length_list = [0]
        fragment_length_freq = np.zeros(1000)

    print(f'Finished {(n*100):.2f}%   ', end='\r')

    return (np.concatenate((np.array([index]), np.array([coverage]), fragment_length_freq), axis=0), fragment_length_list)


def process_chunk(
        bam_file_path, 
        fasta_path, 
        chunk
        ):
    with pysam.AlignmentFile(bam_file_path, "rb") as bam_file, pysam.FastaFile(fasta_path) as fastafile:
        size = len(chunk)
        region_and_fragment_length_list = [per_region(bam_file, fastafile, i/size, index, chrom, start, end) for i, (index, chrom, start, end) in enumerate(chunk.values)]
        regions = [x[0] for x in region_and_fragment_length_list]
        fragment_length_lists = [x[1] for x in region_and_fragment_length_list]
        return (regions, fragment_length_lists)
    

class ReadCounter:
    def __init__(self, config) -> None:
        self.chunk_size = config['DEFAULT']['chunk_size']
        self.chrom = config.get('DEFAULT', 'chrom', fallback=None)
        self.processes = config.getint('DEFAULT', 'processes')

        self.chroms = [str(i) for i in range(1, 23)]

        self.bam_file = config['PATH']['bam_file']
        self.count_file = config['PATH']['count_file']
        self.ref_file = config['PATH']['ref_file']
        self.gc_file = config['PATH']['gc_file']
        self.map_file = config['PATH']['map_file']

        self.outpath = config['PATH']['outpath']
        os.makedirs(self.outpath, exist_ok=True)

        self.name = config['PARAMS']['name']

        self.gc_map = pd.merge(self.parse_wig(self.gc_file, 'gc'), 
                               self.parse_wig(self.map_file, 'map'), 
                               on=['chrom', 'start', 'end'])
        self.gc_map = self.gc_map[self.gc_map['chrom'].isin(self.chroms)]


    def correct_readcount(self, data, col_reads, col='gc', frac_rough=0.03, frac_final=0.3):
        select = data.sample(n=min(len(data), 50000))
        rough = sm.nonparametric.lowess(select[col_reads], select[col], frac=frac_rough)
        i = np.linspace(0, 1, num=1001)
        y_rough = np.interp(i, rough[:, 0], rough[:, 1])
        final = sm.nonparametric.lowess(y_rough, i, frac=frac_final)
        interp_final = lambda x: np.interp(x, final[:, 0], final[:, 1])
        return data[col_reads] / data[col].map(interp_final)
    

    def process_regions(self):
        regions = self.gc_map.merge(pd.DataFrame(self.regions[:,1:], index=self.regions[:,0].astype(int)), left_index=True, right_index=True).rename(columns={0:'reads'})
        regions = regions[(regions['reads'] > 0) & (regions['gc'] > 0)]
        regions = regions[(regions['reads'] > regions['reads'].quantile(0.01)) & (regions['reads'] < regions['reads'].quantile(0.99))]
        regions = regions[(regions['gc'] > regions['gc'].quantile(0.01)) & (regions['gc'] < regions['gc'].quantile(0.99))]
        regions = regions[regions['map'] > 0.8]
        
        regions['gc_corrected'] = self.correct_readcount(regions, col_reads='reads', col='gc')
        upper_quantile = regions['gc_corrected'].quantile(0.99)
        regions = regions[regions['gc_corrected'] < upper_quantile]
        regions['map_corrected'] = self.correct_readcount(regions, col_reads='gc_corrected', col='map')
        regions.loc[regions['map_corrected'] <= 0, 'map_corrected'] = np.nan
        regions['log2'] = np.log2(regions['map_corrected'])
        # center to the median
        regions['log2'] = regions['log2'] - regions['log2'].median()

        return regions
    

    def rd(self, control=False):
        bed = self.gc_map.copy()
        bed = bed.reset_index()
        bed = bed[['index', 'chrom', 'start', 'end']]
        
        if self.chrom: bed = bed[bed.chrom == self.chrom]

        pool = Pool(processes=self.processes)
        bed_chunks = np.array_split(bed, self.processes)
        futures = []

        for chunk in bed_chunks:
            result = pool.apply_async(process_chunk, args=(self.bam_file, self.ref_file, chunk))
            futures.append(result)

        pool.close()
        pool.join()

        results = [future.get() for future in futures]
        regions = np.vstack([x[0] for x in results])

        with open(os.path.join(self.outpath, f'{self.name}.regions.pkl'), 'wb') as f:
            pickle.dump(regions, f)
        
        all_fragment_length_lists = []
        for fragment_length_lists in [x[1] for x in results]:
            for fragment_length_list in fragment_length_lists:
                all_fragment_length_lists.extend(fragment_length_list)
        all_fragment_length_freq = np.bincount(np.array(all_fragment_length_lists))
        all_fragment_length_freq.resize(1000)
        all_fragment_length_freq = all_fragment_length_freq / len(all_fragment_length_lists)

        with open(os.path.join(self.outpath, f'{self.name}.fragment_length_freq.pkl'), 'wb') as f:
            pickle.dump(all_fragment_length_freq, f)
        
        return regions, all_fragment_length_freq
    

    def read_rd(self):
        with open(os.path.join(self.outpath, f'{self.name}.regions.pkl'), 'rb') as f:
            self.regions = pickle.load(f)
        with open(os.path.join(self.outpath, f'{self.name}.fragment_length_freq.pkl'), 'rb') as f:
            self.all_fragment_length_freq = pickle.load(f)
        return self.regions, self.all_fragment_length_freq
    

    def parse_wig(self, filename, value_name) -> pd.DataFrame:
        chunks = []
        chrom = None
        start = None
        step = None

        with open(filename, 'r') as f:
            chunk_data = []
            for line in f:
                line = line.strip()

                if line.startswith('track') or line.startswith('#'):
                    continue

                if line.startswith('fixedStep'):
                    fields = line.split()
                    for field in fields:
                        if field.startswith('chrom='):
                            chrom = field.split('=')[1]
                        elif field.startswith('start='):
                            start = int(field.split('=')[1])
                        elif field.startswith('step='):
                            step = int(field.split('=')[1])
                else:
                    end = start + step - 1
                    chunk_data.append([chrom, start, end, float(line)])
                    start = end + 1

                    if len(chunk_data) == self.chunk_size:
                        chunks.append(pd.DataFrame(chunk_data, columns=['chrom', 'start', 'end', value_name]))
                        chunk_data = []

            if chunk_data:
                chunks.append(pd.DataFrame(chunk_data, columns=['chrom', 'start', 'end', value_name]))

        df = pd.concat(chunks, ignore_index=True)
        df['chrom'] = df['chrom'].astype(str)
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)
        return df
    

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    read_counter = ReadCounter(config=config)

    read_counter.rd()
    read_counter.read_rd()
    print(read_counter.regions)
