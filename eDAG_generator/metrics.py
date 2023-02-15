# All classes related to calculating different types
# of metrics based on either traces or eDAGs
import os
import math
import matplotlib.pyplot as plt
from typing import List, Optional, Set, Dict
from collections import OrderedDict, defaultdict
from instruction_parser import InstructionParser



class ReuseDistance:
    """
    An object representation of the 
    """
    def __init__(self, trace_file: str, parser: InstructionParser) -> None:
        """
        @param trace_file: path to the trace file.
        @param parser: an `InstructionParser` object
        """
        assert(os.path.exists(trace_file))
        self.trace_file = trace_file
        self.reuse_distance = self.__compute_reuse_distance(parser)

    def __compute_reuse_distance(self, parser: InstructionParser) -> Dict[int, int]:
        """
        A private helper function that computes reuse distance 
        of each unique data address by iterating through the give 
        trace file. The reuse distance is calculated with the most naive
        approach where both the time and space complexity is O(n).
        In order for the function to work, an `InstructionParser`
        needs to be passed in as an argument. It will be used to determine
        the virtual addresses that are included in the 
        
        FIXME: Since it is definitely not the most efficient way to do
        this, it might take a considerable amount of time to obtain
        the reuse histograms for large traces.

        References:
        https://roclocality.org/2015/03/18/efficient-computation-of-reuse-distance/
        https://www.cc.gatech.edu/classes/AY2019/cs7260_spring/lecture30.pdf

        Returns the reuse histogram as a dictionary where the keys are
        distances and the values are numbers of occurrences.
        """
        res = defaultdict(int)
        trace = open(self.trace_file, "r")
        
        dist = OrderedDict()
        def get_distance(addr: str) -> int:
            """
            A helper function that returns the reuse distance
            of the given address based on its index in `dist`
            """
            # print(dist)
            if addr not in dist:
                # If address is seen for the first time, 
                # pushes the address to the front of `dist` and
                # returns infinity
                dist[addr] = None
                return math.inf
            # Otherwise, traverses through `dist` and finds the index
            # of the key that matches `addr`
            distance = -1
            for idx, key in enumerate(dist.keys()):
                if key == addr:
                    distance = len(dist) - idx - 1
            # Re-inserts the given address into `dist` so that it is
            # at the very front
            dist.pop(addr)
            dist[addr] = None
            assert(distance >= 0)
            return distance

        for line in trace:
            # Parses each line in the trace file with the given parser
            data_addr = parser.get_load_data_addr(line)
            if data_addr is None:
                continue
            distance = get_distance(data_addr)
            # print(f"[DEBUG] addr: {data_addr}, dist: {distance}")
            res[distance] += 1
        trace.close()
        return res

    
    def plot_histogram(self, save_fig_path: Optional[str] = None) -> None:
        """
        Plots the reuse histogram with matplotlib.
        If `save_fig_path` is not None, the generated histogram
        will be saved to the specified path.
        """
        print(self.reuse_distance)
        # Unzips the key-value pairs in the dictionary into two lists
        reuse_distance = sorted(list(self.reuse_distance.items()),
                                key=lambda p: p[0])
        x, y = list(zip(*reuse_distance))
        x = list(map(lambda elem: str(elem), x))
        plt.figure(figsize=(10, 5))
        plt.bar(x, y)

        # Sets the axis labels
        plt.xlabel("Reuse distance")
        plt.yscale("log")
        plt.ylabel("Number of references")

        _, labels = plt.xticks()
        plt.setp(labels, rotation=90)

        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches="tight")
        else:
            plt.show()
        