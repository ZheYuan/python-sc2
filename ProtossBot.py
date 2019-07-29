#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sc2
from sc2 import position, Result
from sc2.constants import UnitTypeId
import random
import numpy as np
import cv2 as cv
import time
import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import os

HEADLESSS = False
#os.environ["SC2PATH"] = '/starcraftstuff/StarCraftII/'

class ProtossBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165  # 每秒大约165个循环
        self.MAX_WORKERS = 50
        self.MAX_GATEWAY = 5
        self.MAX_GATEWAY_RATIO = 2
        self.do_something_after = 0
        self.train_data = []

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.scout()
        await self.distribute_workers()
        await self.build_workers()  # 建造农民
        await self.build_pylons()  # 建造水晶
        await self.build_assimilators()  # 建造瓦斯气泉
        await self.expand()  # 开分矿
        await self.offensive_force_buildings()  # 建造所需兵营及科技建筑
        await self.build_offensive_force()  # 建造攻击单位
        await self.intel()  # 绘制地图
        await self.attack()  # 攻击

    async def build_workers(self):  # 建造农民
        if (len(self.units(UnitTypeId.NEXUS)) * 16) > len(self.units(UnitTypeId.PROBE)) \
                and len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS:  # 每个矿16个农民，最大50个农民
            for nexus in self.units(UnitTypeId.NEXUS).ready.noqueue:
                if self.can_afford(UnitTypeId.PROBE):
                    await self.do(nexus.train(UnitTypeId.PROBE))

    async def build_pylons(self):  # 建造水晶
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):  # 人口上限<5
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexuses.first)

    async def build_assimilators(self):  # 建造瓦斯气泉
        for nexus in self.units(UnitTypeId.NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)  # 找到基地15方格内的气矿
            for vaspene in vaspenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):  # 确保有足够的矿物资源
                    break
                worker = self.select_build_worker(vaspene.position)  # 分配采气农民
                if worker is None:
                    break
                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vaspene))

    async def expand(self):  # 开分矿
        if self.units(UnitTypeId.NEXUS).amount < (self.iteration / self.ITERATIONS_PER_MINUTE) \
                and self.can_afford(UnitTypeId.NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):  # 建造所需兵营及科技建筑
        # print(self.iteration / self.ITERATIONS_PER_MINUTE)
        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random  # 选择一个Pylon附近建造

            if self.units(UnitTypeId.GATEWAY).ready.exists \
                    and not self.units(UnitTypeId.CYBERNETICSCORE):  # 如果已经有传送门，则建造机械核心
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)  # 机械核心

            elif len(self.units(UnitTypeId.GATEWAY)) < ((self.iteration / self.ITERATIONS_PER_MINUTE) / 2):
                if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon)  # 传送门

            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.units(UnitTypeId.ROBOTICSFACILITY)) < 1:
                    if self.can_afford(UnitTypeId.ROBOTICSFACILITY) \
                            and not self.already_pending(UnitTypeId.ROBOTICSFACILITY):
                        await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)  # 机械工厂，生产哨兵、不朽、巨像

            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.units(UnitTypeId.STARGATE)) < ((self.iteration / self.ITERATIONS_PER_MINUTE) / 2):
                    if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon)  # 星际之门

    async def build_offensive_force(self):  # 建造攻击单位
        for gw in self.units(UnitTypeId.GATEWAY).ready.noqueue:
            if not self.units(UnitTypeId.STALKER).amount > self.units(UnitTypeId.VOIDRAY).amount:
                if self.can_afford(UnitTypeId.STALKER) and self.supply_left > 0:
                    await self.do(gw.train(UnitTypeId.STALKER))  # 追猎

        for sg in self.units(UnitTypeId.STARGATE).ready.noqueue:
            if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(UnitTypeId.VOIDRAY))  # 虚空

    def find_target(self, state):  # 寻找目标
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        if len(self.units(UnitTypeId.VOIDRAY).idle) > 0:
            choice = random.randrange(0, 4)
            target = False
            if self.iteration > self.do_something_after:
                if choice == 0:
                    # no attack
                    wait = random.randrange(20, 165)
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    # attack_unit_closest_nexus
                    if len(self.known_enemy_units) > 0 and self.units(UnitTypeId.NEXUS).empty() is False:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(UnitTypeId.NEXUS)))

                elif choice == 2:
                    # attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    # attack_enemy_start
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(UnitTypeId.VOIDRAY).idle:
                        await self.do(vr.attack(target))
                y = np.zeros(4)
                y[choice] = 1
                print("choice: " + str(y))
                self.train_data.append([y, self.flipped])

    async def intel(self):
        # for game_info: https://github.com/Dentosal/python-sc2/blob/master/sc2/game_info.py#L162
        # print(self.game_info.map_size)
        # flip around. It's y, x when you're dealing with an array.
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR COLOR)]
        '''from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY'''
        draw_dict = {
            UnitTypeId.NEXUS: [15, (0, 255, 0)],
            UnitTypeId.PYLON: [3, (20, 235, 0)],
            UnitTypeId.PROBE: [1, (55, 200, 0)],
            UnitTypeId.ASSIMILATOR: [2, (55, 200, 0)],
            UnitTypeId.GATEWAY: [3, (200, 100, 0)],
            UnitTypeId.CYBERNETICSCORE: [3, (150, 150, 0)],
            UnitTypeId.STARGATE: [5, (255, 0, 0)],
            UnitTypeId.ROBOTICSFACILITY: [5, (215, 155, 0)],
            UnitTypeId.VOIDRAY: [3, (255, 100, 0)],
            # UnitTypeId.OBSERVER: [3, (255, 255, 255)],
        }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_names = ["nexus", "supplydepot", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(UnitTypeId.OBSERVER).ready:
            pos = obs.position
            cv.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / (self.supply_cap + 1e-7)
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(UnitTypeId.VOIDRAY)) / (self.supply_cap - self.supply_left + 1e-7)
        if military_weight > 1.0:
            military_weight = 1.0

        cv.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200),
                3)  # plausible supply (supply/200.0)
        cv.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150),
                3)  # population ratio (supply_left/supply)
        cv.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv.flip(game_data, 0)
        if not HEADLESSS:
            resized = cv.resize(self.flipped, dsize=None, fx=2, fy=2)

            cv.imshow('Intel', resized)
            cv.waitKey(1)



    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    async def scout(self):
        if len(self.units(UnitTypeId.OBSERVER)) > 0:
            scout = self.units(UnitTypeId.OBSERVER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                print("SCOUT: " + str(move_to))
                await self.do(scout.move(move_to))

        else:
            for rf in self.units(UnitTypeId.ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(UnitTypeId.OBSERVER))

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)

        if game_result == Result.Victory:
            np.save("model/train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))


