import os
import sys
import configparser
from multiprocessing import Pool, cpu_count
import pickle

import pandas as pd
import numpy as np
import pysam


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


def process_chunk(bam_file_path, fasta_path, chunk):
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

        with open(f'{self.outpath}.{self.name}.pkl', 'wb') as f:
            pickle.dump(regions, f)
        
        all_fragment_length_lists = []
        for fragment_length_lists in [x[1] for x in results]:
            for fragment_length_list in fragment_length_lists:
                all_fragment_length_lists.extend(fragment_length_list)
        all_fragment_length_freq = np.bincount(np.array(all_fragment_length_lists))
        all_fragment_length_freq.resize(1000)
        all_fragment_length_freq = all_fragment_length_freq / len(all_fragment_length_lists)

        with open(f'{self.outpath}.{self.name}.fragment_lengths.pkl', 'wb') as f:
            pickle.dump(all_fragment_length_freq, f)
        
        return regions, all_fragment_length_freq
    

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

    print(read_counter.gc_map.head())
    read_counter.rd()