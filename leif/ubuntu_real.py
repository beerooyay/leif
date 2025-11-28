"""
ubuntu dialogue corpus parser - REAL DATA

parses the kaggle ubuntu dialogue corpus CSV into lexia format.
this is actual IRC chat data with multiple speakers per conversation.
"""

import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from .data import Lexia, Conversation, LexiaDataset


def simple_tokenize(text: str) -> List[str]:
    """simple whitespace tokenizer"""
    # lowercase and split
    words = text.lower().split()
    # keep only alphanumeric, basic punctuation
    tokens = []
    for word in words:
        word = re.sub(r'[^\w\?\!\.\,]', '', word)
        if word and len(word) < 30:  # skip super long tokens
            tokens.append(word)
    return tokens if tokens else ["<unk>"]


def parse_ubuntu_csv(
    csv_path: str,
    max_conversations: int = 10000,
    min_turns: int = 10,
    max_turns: int = 500,
    min_speakers: int = 3,
) -> Tuple[List[Conversation], Dict[str, int]]:
    """
    parse ubuntu dialogue CSV into conversations.
    
    CSV format: folder,dialogueID,date,from,to,text
    """
    print(f"parsing {csv_path}...")
    
    # group messages by dialogueID
    dialogues = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 500000:  # read up to 500k lines
                break
            
            dialogue_id = row.get('dialogueID', '')
            if not dialogue_id:
                continue
            
            dialogues[dialogue_id].append({
                'from': row.get('from', '').lower().strip(),
                'to': row.get('to', '').lower().strip(),
                'text': row.get('text', ''),
                'date': row.get('date', ''),
            })
    
    print(f"found {len(dialogues)} unique dialogues")
    
    # build vocabulary from all text
    word_counts = defaultdict(int)
    
    for dialogue_id, messages in dialogues.items():
        for msg in messages:
            tokens = simple_tokenize(msg['text'])
            for tok in tokens:
                word_counts[tok] += 1
    
    # filter vocabulary by frequency
    vocab_words = [w for w, c in word_counts.items() if c >= 5]
    vocab_words = sorted(vocab_words)
    
    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    all_tokens = special + vocab_words
    word2id = {w: i for i, w in enumerate(all_tokens)}
    
    print(f"vocabulary size: {len(word2id)}")
    
    # convert dialogues to conversations
    conversations = []
    agent_cache = {}  # map usernames to agent ids per conversation
    
    for dialogue_id, messages in dialogues.items():
        if len(messages) < min_turns or len(messages) > max_turns:
            continue
        
        # check speaker count
        speakers = set(m['from'] for m in messages if m['from'])
        if len(speakers) < min_speakers:
            continue
        
        # reset agent mapping for this conversation
        agent_map = {}
        agent_counter = 0
        
        lexia_list = []
        
        for turn_idx, msg in enumerate(messages):
            sender_name = msg['from']
            receiver_name = msg['to']
            text = msg['text']
            
            if not sender_name or not text:
                continue
            
            # assign agent ids
            if sender_name not in agent_map:
                agent_map[sender_name] = agent_counter
                agent_counter += 1
            
            if receiver_name and receiver_name not in agent_map:
                agent_map[receiver_name] = agent_counter
                agent_counter += 1
            
            sender_id = agent_map[sender_name]
            receiver_id = agent_map.get(receiver_name, sender_id)  # default to self if no receiver
            
            # tokenize
            tokens = simple_tokenize(text)
            token_ids = [word2id.get(tok, word2id["<unk>"]) for tok in tokens]
            
            lexia_list.append(Lexia(
                sender=sender_id,
                receiver=receiver_id,
                conduit=0,
                timestamp=turn_idx,
                tokens=token_ids,
            ))
        
        if len(lexia_list) >= min_turns:
            # cap agent ids to reasonable max
            max_agents = 32
            for lex in lexia_list:
                lex.sender = lex.sender % max_agents
                lex.receiver = lex.receiver % max_agents
            
            conversations.append(Conversation(
                lexia=lexia_list,
                n_agents=min(agent_counter, max_agents),
                n_conduits=1,
            ))
        
        if len(conversations) >= max_conversations:
            break
    
    print(f"created {len(conversations)} conversations")
    
    # stats
    if conversations:
        agent_counts = [conv.n_agents for conv in conversations]
        print(f"agents per conversation: {min(agent_counts)}-{max(agent_counts)} (avg: {sum(agent_counts)/len(agent_counts):.1f})")
    
    return conversations, word2id


def load_ubuntu_real(
    data_dir: str = "data/ubuntu_real/Ubuntu-dialogue-corpus",
    output_dir: str = "data/ubuntu_parsed",
    max_conversations: int = 10000,
    max_seq_len: int = 128,
) -> LexiaDataset:
    """load and parse real ubuntu data from multiple files"""
    
    data_dir = Path(data_dir)
    
    # parse all CSV files
    all_conversations = []
    all_word_counts = defaultdict(int)
    
    csv_files = [
        "dialogueText_196.csv",
        "dialogueText_301.csv",
        "dialogueText.csv",
    ]
    
    for csv_file in csv_files:
        csv_path = data_dir / csv_file
        if csv_path.exists():
            print(f"\nparsing {csv_file}...")
            convs, word2id = parse_ubuntu_csv(
                str(csv_path),
                max_conversations=max_conversations - len(all_conversations),
            )
            all_conversations.extend(convs)
            print(f"  total so far: {len(all_conversations)}")
            
            if len(all_conversations) >= max_conversations:
                break
    
    # rebuild vocabulary from all conversations
    word_counts = defaultdict(int)
    for conv in all_conversations:
        for lex in conv.lexia:
            for tok in lex.tokens:
                word_counts[tok] = word_counts.get(tok, 0) + 1
    
    # use the word2id from the last parse (it's comprehensive enough)
    conversations = all_conversations[:max_conversations]
    
    dataset = LexiaDataset(conversations, word2id, max_seq_len)
    
    # save
    print(f"saving to {output_dir}...")
    dataset.save(output_dir)
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="parse ubuntu dialogue corpus")
    parser.add_argument("--input", type=str, default="data/ubuntu_real/Ubuntu-dialogue-corpus")
    parser.add_argument("--output", type=str, default="data/ubuntu_parsed")
    parser.add_argument("--max_conversations", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    
    args = parser.parse_args()
    
    dataset = load_ubuntu_real(
        data_dir=args.input,
        output_dir=args.output,
        max_conversations=args.max_conversations,
        max_seq_len=args.max_seq_len,
    )
    
    print(f"\ndone. total sequences: {len(dataset)}")
