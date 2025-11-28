"""
synthetic multi-party dialogue generation for leif benchmarking

generates conversations with:
- 5-10 agents per conversation
- explicit turn-taking structure
- direct addressing (@mentions)
- multiple conduits (channels)
- realistic vocabulary and sentence patterns
"""

import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


@dataclass
class Lexia:
    """a single participation event"""
    sender: int
    receiver: int
    conduit: int
    timestamp: int
    tokens: List[int]


@dataclass
class Conversation:
    """a sequence of lexia forming a conversation"""
    lexia: List[Lexia]
    n_agents: int
    n_conduits: int


# vocabulary for synthetic dialogue
GREETINGS = ["hello", "hi", "hey", "greetings", "yo", "sup"]
RESPONSES = ["yes", "no", "maybe", "sure", "okay", "right", "agreed", "disagree"]
QUESTIONS = ["what", "why", "how", "when", "where", "who", "which"]
FILLERS = ["um", "uh", "well", "so", "like", "actually", "basically"]
CONNECTORS = ["and", "but", "or", "because", "since", "although", "however"]
VERBS = ["think", "believe", "know", "see", "want", "need", "have", "make", "do", "say"]
NOUNS = ["thing", "idea", "point", "issue", "problem", "solution", "way", "time", "person"]
ADJECTIVES = ["good", "bad", "new", "old", "big", "small", "important", "different"]
PRONOUNS = ["i", "you", "we", "they", "it", "that", "this"]
ARTICLES = ["the", "a", "an"]
PREPOSITIONS = ["to", "for", "with", "about", "on", "in", "at", "from"]

ALL_WORDS = (
    GREETINGS + RESPONSES + QUESTIONS + FILLERS + CONNECTORS +
    VERBS + NOUNS + ADJECTIVES + PRONOUNS + ARTICLES + PREPOSITIONS
)


def build_vocabulary() -> Tuple[Dict[str, int], Dict[int, str]]:
    """build word to id and id to word mappings"""
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    all_tokens = special_tokens + sorted(set(ALL_WORDS))
    
    word2id = {word: i for i, word in enumerate(all_tokens)}
    id2word = {i: word for word, i in word2id.items()}
    
    return word2id, id2word


def generate_utterance(word2id: Dict[str, int], min_len: int = 2, max_len: int = 12) -> List[int]:
    """generate a random utterance as token ids"""
    length = random.randint(min_len, max_len)
    
    # simple grammar: start with optional filler, then mix of words
    words = []
    
    if random.random() < 0.2:
        words.append(random.choice(FILLERS))
    
    # greeting or question start
    if random.random() < 0.15:
        words.append(random.choice(GREETINGS))
    elif random.random() < 0.2:
        words.append(random.choice(QUESTIONS))
    
    # fill with content
    while len(words) < length:
        category = random.choice([PRONOUNS, VERBS, ARTICLES, NOUNS, ADJECTIVES, PREPOSITIONS, CONNECTORS])
        words.append(random.choice(category))
    
    # response ending
    if random.random() < 0.15:
        words.append(random.choice(RESPONSES))
    
    return [word2id.get(w, word2id["<unk>"]) for w in words[:length]]


def generate_conversation(
    word2id: Dict[str, int],
    n_agents: int = None,
    n_conduits: int = 1,
    n_turns: int = None,
) -> Conversation:
    """
    generate a synthetic multi-party conversation.
    
    args:
        word2id: vocabulary mapping
        n_agents: number of agents (random 5-10 if none)
        n_conduits: number of conduits/channels
        n_turns: number of turns (random 20-50 if none)
    """
    if n_agents is None:
        n_agents = random.randint(5, 10)
    if n_turns is None:
        n_turns = random.randint(20, 50)
    
    lexia_list = []
    timestamp = 0
    
    # track who spoke recently for realistic turn-taking
    recent_speakers = []
    
    for turn in range(n_turns):
        # select sender (avoid immediate self-reply usually)
        if recent_speakers and random.random() < 0.8:
            # respond to someone who spoke recently
            sender = random.choice([i for i in range(n_agents) if i not in recent_speakers[-2:]])
        else:
            sender = random.randint(0, n_agents - 1)
        
        # select receiver
        if recent_speakers and random.random() < 0.7:
            # address someone who spoke recently
            receiver = random.choice(recent_speakers[-3:])
        else:
            # address someone else
            candidates = [i for i in range(n_agents) if i != sender]
            receiver = random.choice(candidates) if candidates else sender
        
        # select conduit
        conduit = random.randint(0, n_conduits - 1)
        
        # generate utterance
        tokens = generate_utterance(word2id)
        
        lexia_list.append(Lexia(
            sender=sender,
            receiver=receiver,
            conduit=conduit,
            timestamp=timestamp,
            tokens=tokens,
        ))
        
        timestamp += 1
        recent_speakers.append(sender)
        if len(recent_speakers) > 5:
            recent_speakers.pop(0)
    
    return Conversation(
        lexia=lexia_list,
        n_agents=n_agents,
        n_conduits=n_conduits,
    )


def conversation_to_tensors(
    conversation: Conversation,
    max_seq_len: int,
    pad_id: int,
    sample_multiparty: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    flatten a conversation into tensors for model input.
    
    each token becomes a position with its lexia's relational coordinates.
    if sample_multiparty=True, tries to find a section with 3+ agents.
    """
    tokens = []
    senders = []
    receivers = []
    conduits = []
    positions = []
    
    # find best starting point if sampling for multiparty
    start_lexia = 0
    if sample_multiparty and len(conversation.lexia) > 10:
        # scan for section with most agent diversity
        best_start = 0
        best_diversity = 0
        window = 50  # look at 50 lexia at a time
        
        for i in range(0, len(conversation.lexia) - window, 10):
            agents = set()
            for lex in conversation.lexia[i:i+window]:
                agents.add(lex.sender)
                agents.add(lex.receiver)
            if len(agents) > best_diversity:
                best_diversity = len(agents)
                best_start = i
        
        start_lexia = best_start
    
    pos = 0
    for lex in conversation.lexia[start_lexia:]:
        for tok in lex.tokens:
            if pos >= max_seq_len:
                break
            tokens.append(tok)
            senders.append(lex.sender)
            receivers.append(lex.receiver)
            conduits.append(lex.conduit)
            positions.append(pos)
            pos += 1
        if pos >= max_seq_len:
            break
    
    # pad to max_seq_len
    seq_len = len(tokens)
    pad_len = max_seq_len - seq_len
    
    tokens = tokens + [pad_id] * pad_len
    senders = senders + [0] * pad_len
    receivers = receivers + [0] * pad_len
    conduits = conduits + [0] * pad_len
    positions = positions + list(range(seq_len, max_seq_len))
    
    return {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "senders": torch.tensor(senders, dtype=torch.long),
        "receivers": torch.tensor(receivers, dtype=torch.long),
        "conduits": torch.tensor(conduits, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "seq_len": seq_len,
    }


class LexiaDataset(Dataset):
    """pytorch dataset for lexia sequences"""
    
    def __init__(
        self,
        conversations: List[Conversation],
        word2id: Dict[str, int],
        max_seq_len: int = 128,
    ):
        self.conversations = conversations
        self.word2id = word2id
        self.max_seq_len = max_seq_len
        self.pad_id = word2id.get("<pad>", 0)
        
        # precompute tensors
        self.data = [
            conversation_to_tensors(conv, max_seq_len, self.pad_id)
            for conv in conversations
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]
    
    @classmethod
    def generate_synthetic(
        cls,
        n_conversations: int = 10000,
        max_seq_len: int = 128,
        n_conduits: int = 2,
    ) -> "LexiaDataset":
        """generate a synthetic dataset"""
        word2id, _ = build_vocabulary()
        
        conversations = [
            generate_conversation(word2id, n_conduits=n_conduits)
            for _ in range(n_conversations)
        ]
        
        return cls(conversations, word2id, max_seq_len)
    
    def save(self, path: str):
        """save dataset to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # save vocabulary
        with open(path / "vocab.json", "w") as f:
            json.dump(self.word2id, f)
        
        # save conversations as json
        convs_data = []
        for conv in self.conversations:
            convs_data.append({
                "n_agents": conv.n_agents,
                "n_conduits": conv.n_conduits,
                "lexia": [
                    {
                        "sender": int(lex.sender),
                        "receiver": int(lex.receiver),
                        "conduit": int(lex.conduit),
                        "timestamp": int(lex.timestamp),
                        "tokens": [int(tok) for tok in lex.tokens],
                    }
                    for lex in conv.lexia
                ]
            })
        
        with open(path / "conversations.json", "w") as f:
            json.dump(convs_data, f)
    
    @classmethod
    def load(cls, path: str, max_seq_len: int = 128) -> "LexiaDataset":
        """load dataset from disk"""
        path = Path(path)
        
        with open(path / "vocab.json") as f:
            word2id = json.load(f)
        
        with open(path / "conversations.json") as f:
            convs_data = json.load(f)
        
        conversations = []
        for cd in convs_data:
            lexia = [
                Lexia(
                    sender=lex["sender"],
                    receiver=lex["receiver"],
                    conduit=lex["conduit"],
                    timestamp=lex["timestamp"],
                    tokens=lex["tokens"],
                )
                for lex in cd["lexia"]
            ]
            conversations.append(Conversation(
                lexia=lexia,
                n_agents=cd["n_agents"],
                n_conduits=cd["n_conduits"],
            ))
        
        return cls(conversations, word2id, max_seq_len)


def generate_synthetic_dialogue(
    output_path: str,
    n_conversations: int = 10000,
    max_seq_len: int = 128,
):
    """generate and save synthetic dialogue dataset"""
    print(f"generating {n_conversations} synthetic conversations...")
    dataset = LexiaDataset.generate_synthetic(n_conversations, max_seq_len)
    
    print(f"saving to {output_path}...")
    dataset.save(output_path)
    
    print(f"done. vocabulary size: {len(dataset.word2id)}")
    print(f"total sequences: {len(dataset)}")
    
    # compute statistics
    total_tokens = sum(d["seq_len"] for d in dataset.data)
    print(f"total tokens: {total_tokens}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="generate synthetic dialogue data")
    parser.add_argument("--output", type=str, default="data/synthetic")
    parser.add_argument("--n_conversations", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    
    args = parser.parse_args()
    
    generate_synthetic_dialogue(
        args.output,
        args.n_conversations,
        args.max_seq_len,
    )
