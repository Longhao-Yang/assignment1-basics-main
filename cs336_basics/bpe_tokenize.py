import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_chunk(
    tasks
) -> dict[str, int]:
    input_path: str | os.PathLike = tasks[0]
    special_tokens: list[str] = tasks[1]
    start, end = tasks[2]
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        
        pattern = "|".join(re.escape(token) for token in special_tokens)
        documents = re.split(pattern, chunk)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        mp = {}
        for text in documents:
            for word in re.finditer(PAT, text):
                wo = word.group()
                if wo in mp:
                    mp[wo] += 1
                else:
                    mp[wo] = 1
        return mp
                    

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        
        tasks = list((input_path, special_tokens, i) for i in zip(boundaries[:-1], boundaries[1:]))
        
        bytes_pair_count: dict[tuple[bytes, bytes], int] = {}
        bytes_map: dict[str, int] = {}
        
        
        bytes_pos:dict[tuple[bytes, bytes], set] = {}
        with ProcessPoolExecutor(8) as executor:
            for mp in tqdm(executor.map(process_chunk, tasks), "chunk进度"):
                for word, cnt in mp.items():
                    if word in bytes_map:
                        bytes_map[word] += cnt
                    else:
                        bytes_map[word] = cnt
        bytes_list: list[tuple[list[bytes], int]] = [([bytes([p]) for p in i.encode()], j) for i, j in bytes_map.items()]
                                  
        for idx, (word, cnt) in enumerate(bytes_list):
            n = len(word)
            for i in range(n - 1):
                if (word[i], word[i + 1]) in bytes_pos:
                    bytes_pos[(word[i], word[i + 1])].add(idx) 
                else:
                    bytes_pos[(word[i], word[i + 1])] = {idx}
                    
                if (word[i], word[i + 1]) in bytes_pair_count:
                    bytes_pair_count[(word[i], word[i + 1])] += cnt
                else:
                    bytes_pair_count[(word[i], word[i + 1])] = cnt
        voc: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        now_cnt = 255
        merges = []       
        
        for _ in range(vocab_size - 257):     
            mx = -1
            pair_update: tuple = ()
            for pair, cnt in bytes_pair_count.items():
                if cnt > mx:
                    mx = cnt
                    pair_update = pair
                elif cnt == mx:
                    if pair > pair_update:
                        pair_update = pair
            if mx <= 1:
                break            
            merges.append(pair_update)
            now_cnt += 1
            voc[now_cnt] = pair_update[0] + pair_update[1]
            
            s = list(bytes_pos[pair_update])
            for idx in s:
                word, cnt = bytes_list[idx]

                new_word = []
                
                tmp = 0
                while tmp < len(word):
                    if (tmp < len(word) - 1) and (word[tmp], word[tmp + 1])  == pair_update:
                        new_word.append(voc[now_cnt])
                        tmp += 2
                    else:
                        new_word.append(word[tmp])
                        tmp += 1
                bytes_list[idx] = (new_word, cnt)        
                n = len(word)
                for i in range(n - 1):
                    bytes_pair_count[(word[i], word[i + 1])] -= cnt
                    bytes_pos[(word[i], word[i + 1])].discard(idx)
                n = len(new_word)
                for i in range(n - 1):
                    if (new_word[i], new_word[i + 1]) in bytes_pair_count:
                        bytes_pair_count[(new_word[i], new_word[i + 1])] += cnt
                    else:
                        bytes_pair_count[(new_word[i], new_word[i + 1])] = cnt
                    if (new_word[i], new_word[i + 1]) in bytes_pos:

                        bytes_pos[(new_word[i], new_word[i + 1])].add(idx)
                    else:
                        bytes_pos[(new_word[i], new_word[i + 1])] = {idx}
        voc[now_cnt + 1] = "<|endoftext|>".encode()
        return (voc, merges)                
                
            

        