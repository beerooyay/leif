"""
ubuntu dialogue corpus loader

converts ubuntu irc logs into lexia format for leif training.
each message becomes a lexia with:
- sender: the username
- receiver: inferred from @mentions or previous speaker
- conduit: channel (always 0 for ubuntu)
- timestamp: message index
- tokens: tokenized message text
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from datasets import load_dataset

from .data import Lexia, Conversation, LexiaDataset, build_vocabulary, ALL_WORDS


def extract_mention(text: str) -> Optional[str]:
    """extract @username or username: at start of message"""
    # pattern: @username or username: at start
    match = re.match(r'^@?(\w+)[:\s]', text)
    if match:
        return match.group(1).lower()
    return None


def simple_tokenize(text: str, word2id: Dict[str, int]) -> List[int]:
    """simple whitespace tokenizer with unknown handling"""
    words = text.lower().split()
    tokens = []
    for word in words:
        # strip punctuation
        word = re.sub(r'[^\w]', '', word)
        if word:
            tokens.append(word2id.get(word, word2id.get("<unk>", 1)))
    return tokens if tokens else [word2id.get("<unk>", 1)]


def build_ubuntu_vocabulary(dialogues: List[List[dict]], min_freq: int = 5) -> Tuple[Dict[str, int], Dict[int, str]]:
    """build vocabulary from ubuntu dialogues"""
    word_counts = defaultdict(int)
    
    for dialogue in dialogues:
        for turn in dialogue:
            text = turn.get("text", turn.get("utterance", ""))
            words = text.lower().split()
            for word in words:
                word = re.sub(r'[^\w]', '', word)
                if word:
                    word_counts[word] += 1
    
    # filter by frequency
    vocab_words = [w for w, c in word_counts.items() if c >= min_freq]
    vocab_words = sorted(vocab_words)
    
    # add special tokens
    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    all_tokens = special + vocab_words
    
    word2id = {w: i for i, w in enumerate(all_tokens)}
    id2word = {i: w for w, i in word2id.items()}
    
    return word2id, id2word


def parse_ubuntu_dialogue(
    dialogue: List[dict],
    word2id: Dict[str, int],
    agent2id: Dict[str, int],
) -> Conversation:
    """
    convert a single ubuntu dialogue to lexia format.
    
    dialogue is a list of turns, each with 'speaker' and 'text'/'utterance'
    """
    lexia_list = []
    recent_speakers = []
    
    for i, turn in enumerate(dialogue):
        # get speaker
        speaker = turn.get("speaker", turn.get("actor", f"user_{i}"))
        speaker = speaker.lower() if speaker else f"user_{i}"
        
        # get or assign agent id
        if speaker not in agent2id:
            agent2id[speaker] = len(agent2id)
        sender_id = agent2id[speaker]
        
        # get text
        text = turn.get("text", turn.get("utterance", ""))
        if not text:
            continue
        
        # infer receiver from @mention or previous speaker
        mention = extract_mention(text)
        if mention and mention in agent2id:
            receiver_id = agent2id[mention]
        elif recent_speakers:
            # default to previous speaker
            receiver_id = recent_speakers[-1]
        else:
            receiver_id = sender_id  # talking to self/channel
        
        # tokenize
        tokens = simple_tokenize(text, word2id)
        
        lexia_list.append(Lexia(
            sender=sender_id,
            receiver=receiver_id,
            conduit=0,
            timestamp=i,
            tokens=tokens,
        ))
        
        recent_speakers.append(sender_id)
        if len(recent_speakers) > 5:
            recent_speakers.pop(0)
    
    return Conversation(
        lexia=lexia_list,
        n_agents=len(agent2id),
        n_conduits=1,
    )


def load_ubuntu_dataset(
    max_dialogues: int = 10000,
    max_seq_len: int = 128,
    min_turns: int = 5,
) -> LexiaDataset:
    """
    load ubuntu dialogue corpus and convert to lexia format.
    
    uses huggingface datasets to download.
    """
    print("loading dialogue corpus from huggingface...")
    
    # use datasets that work with current huggingface api
    try:
        # multi-party chat dataset
        print("trying multi_woz_v22...")
        dataset = load_dataset("pfb30/multi_woz_v22", split="train")
    except Exception as e1:
        print(f"multi_woz failed: {e1}")
        try:
            # persona chat - good multi-turn dialogue
            print("trying bavard/personachat_truecased...")
            dataset = load_dataset("bavard/personachat_truecased", split="train")
        except Exception as e2:
            print(f"personachat failed: {e2}")
            try:
                # empathetic dialogues
                print("trying empathetic_dialogues...")
                dataset = load_dataset("empathetic_dialogues", split="train")
            except Exception as e3:
                print(f"empathetic_dialogues failed: {e3}")
                # blended skill talk
                print("trying blended_skill_talk...")
                dataset = load_dataset("blended_skill_talk", split="train")
    
    print(f"loaded {len(dataset)} examples")
    
    # extract dialogues
    dialogues = []
    for i, example in enumerate(dataset):
        if i >= max_dialogues * 2:  # get extra to filter
            break
        
        # handle different dataset formats
        turns = []
        
        if "turns" in example:
            # multi_woz format
            for j, turn in enumerate(example["turns"]):
                speaker = turn.get("speaker", f"user_{j%2}")
                text = turn.get("utterance", turn.get("text", ""))
                if text:
                    turns.append({"speaker": speaker, "text": text})
        elif "dialog" in example:
            # daily_dialog format: list of utterances
            turns = [{"speaker": f"user_{j%2}", "text": utt} for j, utt in enumerate(example["dialog"])]
        elif "utterances" in example:
            # personachat format
            for j, utt in enumerate(example["utterances"]):
                if isinstance(utt, dict):
                    turns.append({"speaker": f"user_{j%2}", "text": utt.get("text", str(utt))})
                else:
                    turns.append({"speaker": f"user_{j%2}", "text": str(utt)})
        elif "conv_id" in example and "utterance" in example:
            # empathetic dialogues - single utterance per row, need to group
            turns.append({"speaker": example.get("speaker_idx", 0), "text": example["utterance"]})
        elif "previous_utterance" in example and "free_messages" in example:
            # blended skill talk - interleave free and guided messages
            prev = example.get("previous_utterance", [])
            free = example.get("free_messages", [])
            guided = example.get("guided_messages", [])
            
            # add previous utterances
            for j, utt in enumerate(prev):
                if utt:
                    turns.append({"speaker": f"user_{j%2}", "text": utt})
            
            # interleave free and guided messages
            for j in range(max(len(free), len(guided))):
                if j < len(free) and free[j]:
                    turns.append({"speaker": "user_0", "text": free[j]})
                if j < len(guided) and guided[j]:
                    turns.append({"speaker": "user_1", "text": guided[j]})
        elif "context" in example and "response" in example:
            # context-response format
            context = example["context"] if isinstance(example["context"], list) else [example["context"]]
            turns = [{"speaker": f"user_{j%2}", "text": utt} for j, utt in enumerate(context)]
            turns.append({"speaker": "user_1", "text": example["response"]})
        elif "messages" in example:
            turns = example["messages"]
        
        if not turns:
            continue
        
        if len(turns) >= min_turns:
            dialogues.append(turns)
        
        if len(dialogues) >= max_dialogues:
            break
    
    print(f"extracted {len(dialogues)} dialogues with >= {min_turns} turns")
    
    # build vocabulary
    print("building vocabulary...")
    word2id, id2word = build_ubuntu_vocabulary(dialogues)
    print(f"vocabulary size: {len(word2id)}")
    
    # convert to lexia format
    print("converting to lexia format...")
    agent2id = {}
    conversations = []
    
    for dialogue in dialogues:
        conv = parse_ubuntu_dialogue(dialogue, word2id, agent2id)
        if conv.lexia:
            conversations.append(conv)
    
    print(f"created {len(conversations)} conversations")
    print(f"total agents seen: {len(agent2id)}")
    
    # create dataset
    # need to limit agent ids to n_agents for embedding
    max_agents = 64
    for conv in conversations:
        for lex in conv.lexia:
            lex.sender = lex.sender % max_agents
            lex.receiver = lex.receiver % max_agents
    
    return LexiaDataset(conversations, word2id, max_seq_len)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="load ubuntu dialogue corpus")
    parser.add_argument("--output", type=str, default="data/ubuntu")
    parser.add_argument("--max_dialogues", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    
    args = parser.parse_args()
    
    dataset = load_ubuntu_dataset(
        max_dialogues=args.max_dialogues,
        max_seq_len=args.max_seq_len,
    )
    
    print(f"\nsaving to {args.output}...")
    dataset.save(args.output)
    
    print(f"done. total sequences: {len(dataset)}")
