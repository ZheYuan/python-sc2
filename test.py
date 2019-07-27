#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId
import random


class SentdeBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()

    async def build_workers(self):  # 建造worker
        if (len(self.units(UnitTypeId.NEXUS)) * 16) > len(self.units(UnitTypeId.PROBE)) and len(
                self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(UnitTypeId.NEXUS).ready.noqueue:
                if self.can_afford(UnitTypeId.PROBE):
                    await self.do(nexus.train(UnitTypeId.PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(UnitTypeId.NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vaspene))

    async def expand(self):
        if self.units(UnitTypeId.NEXUS).amount < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(
                UnitTypeId.NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        # print(self.iteration / self.ITERATIONS_PER_MINUTE)
        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random

            if self.units(UnitTypeId.GATEWAY).ready.exists and not self.units(UnitTypeId.CYBERNETICSCORE):
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

            elif len(self.units(UnitTypeId.GATEWAY)) < ((self.iteration / self.ITERATIONS_PER_MINUTE) / 2):
                if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon)

            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.units(UnitTypeId.STARGATE)) < ((self.iteration / self.ITERATIONS_PER_MINUTE) / 2):
                    if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon)

    async def build_offensive_force(self):
        for gw in self.units(UnitTypeId.GATEWAY).ready.noqueue:
            if not self.units(UnitTypeId.STALKER).amount > self.units(UnitTypeId.VOIDRAY).amount:
                if self.can_afford(UnitTypeId.STALKER) and self.supply_left > 0:
                    await self.do(gw.train(UnitTypeId.STALKER))

        for sg in self.units(UnitTypeId.STARGATE).ready.noqueue:
            if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(UnitTypeId.VOIDRAY))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        aggressive_units = {UnitTypeId.STALKER: [15, 5],
                            UnitTypeId.VOIDRAY: [8, 3]}

        for UNIT in aggressive_units:
            if self.units(UNIT).amount > aggressive_units[UNIT][0] \
                    and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(self.find_target(self.state)))

            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))


run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Hard)
], realtime=False)
