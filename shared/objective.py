"""
objective.py - Shared cost function for roommate matching
CS4100 AI Project

everyone imports from here so we're all using the same 
cost function and student data structure
"""


class Student:
    """stores a student's preferences for matching"""

    def __init__(self, name, sleep, clean, noise, roommate_count, room_features):
        # sleep: 1 = early bird, 5 = night owl
        # clean: 1 = messy, 5 = super clean
        # noise: 1 = needs quiet, 5 = doesnt care about noise
        # roommate_count: how many roommates they want (1-4)
        # room_features: set of things they want like "AC", "private_bath", etc
        self.name = name
        self.sleep = sleep
        self.clean = clean
        self.noise = noise
        self.roommate_count = roommate_count
        self.room_features = set(room_features)

    def __repr__(self):
        return f"Student({self.name})"


def pair_cost(s1, s2):
    """
    how compatible are two students? lower = better match
    
    we just take the absolute difference of each preference
    and add them up. if two students are identical on everything
    the cost is 0 (perfect roommates)
    
    cost = |sleep1 - sleep2| + |clean1 - clean2| + |noise1 - noise2|
         + |roommate_count1 - roommate_count2| 
         + |num_features1 - num_features2|
    """
    return (
        abs(s1.sleep - s2.sleep)
        + abs(s1.clean - s2.clean)
        + abs(s1.noise - s2.noise)
        + abs(s1.roommate_count - s2.roommate_count)
        + abs(len(s1.room_features) - len(s2.room_features))
    )


def total_cost(pairs):
    """
    given a list of (student, student) pairs, compute the 
    total cost across all pairs. this is what all our 
    algorithms are trying to minimize
    """
    total = 0
    for s1, s2 in pairs:
        total += pair_cost(s1, s2)
    return total
