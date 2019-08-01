#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ProtossBot import ProtossBot
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer

if __name__ == '__main__':
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, ProtossBot()),
        Computer(Race.Terran, Difficulty.Hard)
    ], realtime=False)
