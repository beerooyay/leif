"""
multi-party synthetic dialogue generator

creates realistic multi-party conversations where relational structure matters.
this is where leif should shine - not in dyadic ping-pong.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from .data import Lexia, Conversation, LexiaDataset


@dataclass
class Agent:
    """a conversational agent with personality and preferences"""
    id: int
    name: str
    topics: List[str]  # preferred topics
    friends: Set[int]  # agents they talk to more
    talkativeness: float  # 0-1, how likely to speak
    formality: float  # 0-1, formal vs casual


class MultiPartyConversation:
    """generates realistic multi-party dialogues"""
    
    def __init__(self, agents: List[Agent], max_turns: int = 50):
        self.agents = {a.id: a for a in agents}
        self.max_turns = max_turns
        self.topics = ["tech", "sports", "music", "movies", "food", "travel", "work", "school"]
        self.current_topic = random.choice(self.topics)
        self.conversation_history = []
        
    def _select_speaker(self, last_speaker: int, last_receiver: int) -> int:
        """choose who speaks next based on conversation dynamics"""
        candidates = list(self.agents.keys())
        
        # bias towards not repeating the same speaker
        if last_speaker != -1:
            weights = []
            for agent_id in candidates:
                agent = self.agents[agent_id]
                weight = agent.talkativeness
                
                # reduce weight if they just spoke
                if agent_id == last_speaker:
                    weight *= 0.3
                
                # increase weight if they were just addressed
                if agent_id == last_receiver:
                    weight *= 2.0
                
                # increase weight if they're friends with last speaker
                if last_speaker in agent.friends:
                    weight *= 1.5
                
                weights.append(weight)
            
            # normalize weights
            total = sum(weights)
            weights = [w/total for w in weights]
            
            return np.random.choice(candidates, p=weights)
        
        # first speaker - pick most talkative
        return max(candidates, key=lambda x: self.agents[x].talkativeness)
    
    def _select_receiver(self, speaker: int, last_speaker: int, last_receiver: int) -> int:
        """choose who this message is addressed to"""
        candidates = [a for a in self.agents.keys() if a != speaker]
        
        # bias towards continuing conversation
        if last_speaker != -1:
            weights = []
            for agent_id in candidates:
                agent = self.agents[agent_id]
                
                # high weight if they just spoke (replying)
                if agent_id == last_speaker:
                    weight = 3.0
                # high weight if they were just addressed
                elif agent_id == last_receiver:
                    weight = 2.0
                else:
                    weight = 1.0
                
                # adjust for friendship
                if agent_id in self.agents[speaker].friends:
                    weight *= 1.5
                
                weights.append(weight)
            
            # normalize
            total = sum(weights)
            weights = [w/total for w in weights]
            
            return np.random.choice(candidates, p=weights)
        
        # random choice for first message
        return random.choice(candidates)
    
    def _generate_message(self, speaker: Agent, receiver: int, context: List[str]) -> str:
        """generate a message based on speaker personality and context"""
        # simple template-based generation
        templates = [
            "{speaker_name}: hey @{receiver_name}, what do you think about {topic}?",
            "{speaker_name}: @{receiver_name}, that's interesting. i think {topic} is {opinion}.",
            "{speaker_name}: speaking of {topic}, has anyone tried {example}?",
            "{speaker_name}: @{receiver_name}, i disagree about {topic}. actually {counterpoint}.",
            "{speaker_name}: guys, let's talk about {topic} for a sec.",
            "{speaker_name}: @{receiver_name}, you mentioned {topic} earlier...",
        ]
        
        template = random.choice(templates)
        
        # fill template
        speaker_name = speaker.name
        receiver_name = self.agents[receiver].name
        topic = self.current_topic
        
        opinions = ["great", "terrible", "okay", "amazing", "overrated"]
        examples = ["the new thing", "that old method", "something different"]
        counterpoints = ["it's better this way", "we should consider alternatives", "the data shows otherwise"]
        
        opinion = random.choice(opinions)
        example = random.choice(examples)
        counterpoint = random.choice(counterpoints)
        
        message = template.format(
            speaker_name=speaker_name,
            receiver_name=receiver_name,
            topic=topic,
            opinion=opinion,
            example=example,
            counterpoint=counterpoint
        )
        
        return message
    
    def generate(self) -> Conversation:
        """generate a multi-party conversation"""
        lexia_list = []
        
        last_speaker = -1
        last_receiver = -1
        context = []
        
        for turn in range(self.max_turns):
            # occasionally switch topics
            if random.random() < 0.1:
                self.current_topic = random.choice(self.topics)
            
            # select participants
            speaker_id = self._select_speaker(last_speaker, last_receiver)
            receiver_id = self._select_receiver(speaker_id, last_speaker, last_receiver)
            
            speaker = self.agents[speaker_id]
            
            # generate message
            message = self._generate_message(speaker, receiver_id, context)
            
            # tokenize (simple split)
            words = message.lower().replace("@", "").replace(":", "").split()
            tokens = [word for word in words if word.isalpha() or word in ["?", "!", ".", ","]]
            
            if not tokens:
                tokens = ["um"]
            
            lexia_list.append(Lexia(
                sender=speaker_id,
                receiver=receiver_id,
                conduit=0,  # single channel
                timestamp=turn,
                tokens=tokens,
            ))
            
            # update state
            last_speaker = speaker_id
            last_receiver = receiver_id
            context.append(message)
            
            # occasionally add a third person jumping in
            if random.random() < 0.2 and len(self.agents) > 2:
                interjector = random.choice([a for a in self.agents.keys() if a not in [speaker_id, receiver_id]])
                interjector_agent = self.agents[interjector]
                
                interjection = f"{interjector_agent.name}: wait, about {self.current_topic}..."
                words = interjection.lower().replace(":", "").split()
                tokens = [word for word in words if word.isalpha()]
                
                lexia_list.append(Lexia(
                    sender=interjector,
                    receiver=speaker_id,  # addressing the current speaker
                    conduit=0,
                    timestamp=turn,
                    tokens=tokens,
                ))
                
                last_speaker = interjector
        
        return Conversation(
            lexia=lexia_list,
            n_agents=len(self.agents),
            n_conduits=1,
        )


def create_agents(n_agents: int) -> List[Agent]:
    """create a diverse set of agents"""
    topics = ["tech", "sports", "music", "movies", "food", "travel", "work", "school"]
    names = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry"]
    
    agents = []
    for i in range(min(n_agents, len(names))):
        # give each agent 2-3 preferred topics
        agent_topics = random.sample(topics, random.randint(2, 3))
        
        # create friendship network (small world)
        friends = set()
        for j in range(n_agents):
            if i != j and random.random() < 0.3:  # 30% chance of friendship
                friends.add(j)
        
        agent = Agent(
            id=i,
            name=names[i],
            topics=agent_topics,
            friends=friends,
            talkativeness=random.uniform(0.3, 1.0),
            formality=random.uniform(0.2, 0.8),
        )
        agents.append(agent)
    
    return agents


def generate_multiparty_dataset(
    n_conversations: int = 1000,
    n_agents_range: Tuple[int, int] = (3, 8),
    max_turns: int = 64,
    max_seq_len: int = 128,
) -> LexiaDataset:
    """generate a dataset of multi-party conversations"""
    print(f"generating {n_conversations} multi-party conversations...")
    
    conversations = []
    word_counts = defaultdict(int)
    
    for i in range(n_conversations):
        # random number of agents for this conversation
        n_agents = random.randint(*n_agents_range)
        agents = create_agents(n_agents)
        
        # generate conversation
        conv_gen = MultiPartyConversation(agents, max_turns)
        conv = conv_gen.generate()
        
        if conv.lexia:
            conversations.append(conv)
            
            # count words for vocabulary
            for lex in conv.lexia:
                for word in lex.tokens:
                    word_counts[word] += 1
        
        if (i + 1) % 100 == 0:
            print(f"  generated {i + 1}/{n_conversations} conversations")
    
    # build vocabulary from word counts
    print("building vocabulary...")
    # filter by frequency
    vocab_words = [w for w, c in word_counts.items() if c >= 2]
    vocab_words = sorted(vocab_words)
    
    # add special tokens
    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    all_tokens = special + vocab_words
    
    word2id = {w: i for i, w in enumerate(all_tokens)}
    id2word = {i: w for w, i in word2id.items()}
    print(f"vocabulary size: {len(word2id)}")
    
    # convert tokens to ids
    for conv in conversations:
        for lex in conv.lexia:
            lex.tokens = [word2id.get(word, word2id.get("<unk>", 1)) for word in lex.tokens]
    
    return LexiaDataset(conversations, word2id, max_seq_len)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="generate multi-party dialogue dataset")
    parser.add_argument("--output", type=str, default="data/multiparty")
    parser.add_argument("--n_conversations", type=int, default=1000)
    parser.add_argument("--min_agents", type=int, default=3)
    parser.add_argument("--max_agents", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    
    args = parser.parse_args()
    
    dataset = generate_multiparty_dataset(
        n_conversations=args.n_conversations,
        n_agents_range=(args.min_agents, args.max_agents),
        max_turns=args.max_turns,
        max_seq_len=args.max_seq_len,
    )
    
    print(f"\nsaving to {args.output}...")
    dataset.save(args.output)
    
    print(f"done. total sequences: {len(dataset)}")
    
    # show some stats
    agent_counts = [conv.n_agents for conv in dataset.conversations]
    print(f"agents per conversation: {min(agent_counts)}-{max(agent_counts)} (avg: {np.mean(agent_counts):.1f})")
