# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:11:50 2019

@author: Callon
"""

import csv
import os.path
from enum import Enum

team_name_dic = {
        "Anaheim Ducks" : "ANH",
        "Arizona Coyotes" : "ARI",
        "Boston Bruins" : "BOS",
        "Buffalo Sabres" : "BUF",
        "Carolina Hurricanes" : "CAR",
        "Calgary Flames" : "CGY",
        "Chicago Blackhawks" : "CHI",
        "Columbus Blue Jackets" : "CLS",
        "Colorado Avalanche" : "COL",
        "Dallas Stars" : "DAL",
        "Detroit Red Wings" : "DET",
        "Edmonton Oilers" : "EDM",
        "Florida Panthers" : "FLA",
        "Los Angeles Kings" : "LA",
        "Minnesota Wild" : "MIN",
        "Montreal Canadiens" : "MON",
        "Nashville Predators" : "NSH",
        "New Jersey Devils" : "NJ",
        "New York Islanders" : "NYI",
        "New York Rangers" : "NYR",
        "Ottawa Senators" : "OTT",
        "Philadelphia Flyers" : "PHI",
        "Pittsburgh Penguins" : "PIT",
        "San Jose Sharks" : "SJ",
        "St. Louis Blues" : "STL",
        "Tampa Bay Lightning" : "TB",
        "Toronto Maple Leafs" : "TOR",
        "Vancouver Canucks" : "VAN",
        "Vegas Golden Knights" : "VGK",
        "Winnipeg Jets" : "WPG",
        "Washington Capitals" : "WAS"
        }

class Positions(Enum):
    RW = 0
    LW = 1
    C = 2
    D = 3

class FeatureVectors:
    def __init__(self, gameDate, isTrain):
        self.game_date = gameDate
        self.isTrain = isTrain
 
        self.teams_playing = []
        self.players = []
        self.games = []
        self.team_stats = []


    def generateFeatureVectors(self):
        '''
        Reading in schedule for specified day
        '''
        with open('2019_2020_NHL_Schedule.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if row[0] == self.game_date:
                    self.teams_playing.append(team_name_dic[row[2]])
                    self.teams_playing.append(team_name_dic[row[3]])
                    self.games.append([team_name_dic[row[2]], team_name_dic[row[3]]])
            print(f'Processed {line_count} Games.')
        
        '''
        Finding all players playing on given day
        '''
        with open('nhl-stats-2019_11_13.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if row[1] in self.teams_playing:
                    self.players.append(row)
            print(f'Processed {line_count - 2} players.')
        
        '''
        Pulling in most up to date team-based stats
        '''
        with open('standings_2019_11_13.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    row[1] = team_name_dic[row[1]]
                if row[1] in self.teams_playing:
                    self.team_stats.append(row)
                line_count += 1
            print(f'Processed {line_count-1} teams.')
        
        '''
        Finding player data for last game played
        '''
        players_lastgame = [] 
        for player in self.players:
            team = player[1]
            last_game_date = ""
            with open('2019_2020_NHL_Schedule.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if (team == team_name_dic[row[2]] or team == team_name_dic[row[3]]) and row[0] < self.game_date:
                        last_game_date = row[0]
            
            stats_file = 'nhl_stats_daily_' + last_game_date + '.csv'
        
            if (os.path.isfile(stats_file)):
                with open(stats_file) as csv_file2:
                    csv_reader2 = csv.reader(csv_file2, delimiter=',')
                    line_count = 0;
                    prev_game_dat = []
                    for row in csv_reader2:
                        if line_count != 0 and line_count != 1:
                            row[1] = row[1].split('\\')[0]
                            if row[1] == player[0]:
                                prev_game_dat = row
                        line_count += 1
                    if prev_game_dat == []:
                        players_lastgame.append(["null"])
                    else:
                        players_lastgame.append(prev_game_dat)
            else:
                players_lastgame.append(["null"])
            
        '''
        Setting up features vector.
        Data will be formatted as:
            [Name, pos (num), G(season), A(season), +/-(season), PIM(season), PPP(season), SOG(season), 
            BLK(season), G(last game), A(last game), +/-(last game), PIM(last game), PPP(last game), SOG(last game), 
            BLK(last game), Team RK, Opponent RK]
        
        New format:
            [Name, G(season), A(season), +/-(season), PIM(season), PPP(season), SOG(season), 
            BLK(season), G(last game), A(last game), +/-(last game), PIM(last game), PPP(last game), SOG(last game), 
            BLK(last game)]
        '''
        print(self.games)
        input_data = []
        last_game_iter = 0
        for player in self.players:
            opponent = ""
            opponent_stats = []
            team_stats1 = []
            for game in self.games:
                if player[1] in game:
                    if player[1] == game[0]:
                        opponent = game[1];
                    else:
                        opponent = game[0];
            for team in self.team_stats:
                if team[1] == opponent:
                    opponent_stats = team
                elif team[1] == player[1]:
                    team_stats1 = team
        
            if player[2] == "RW":
                player[2] = Positions.RW.value;
            elif player[2] == "LW":
                player[2] = Positions.LW.value;
            elif player[2] == "C":
                player[2] = Positions.C.value;
            elif player[2] == "D":
                player[2] = Positions.D.value;
                
            player_lg = players_lastgame[last_game_iter]
            last_game_iter += 1
            if player_lg[0] == "null":
                input_data.append([player[0], int(player[4]), int(player[5]), int(player[7]),
                                   int(player[8]), int(player[11]) + int(player[12]), int(player[9]), int(player[16]), 
                                   0, 0, 0, 0, 0, 0, 0,
                                   int(team_stats1[0]), int(opponent_stats[0])])
            else:
                input_data.append([player[0], int(player[4]), int(player[5]), int(player[7]),
                                   int(player[8]), int(player[11]) + int(player[12]), int(player[9]), int(player[16]), 
                                   int(player_lg[8]), int(player_lg[9]), int(player_lg[11]), int(player_lg[12]), 
                                   int(player_lg[14]) + int(player_lg[18]), int(player_lg[20]), int(player_lg[25]),
                                   int(team_stats1[0]), int(opponent_stats[0])])
                
        if self.isTrain:
            '''
            Get the fantasy point totals that actually occured this night
            '''
            actualResults = []
            game_night = stats_file = 'nhl_stats_daily_' + self.game_date + '.csv'
            for player in self.players:
                with open(game_night) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    player_found = False
                    for row in csv_reader:
                        if line_count != 0:
                            row[1] = row[1].split('\\')[0]
                            if row[1] == player[0]:
                                player_found = True
                                total = int(row[8]) + int(row[9]) + int(row[11]) + int(row[12]) + int(row[14]) + int(row[18]) + int(row[20]) +int(row[25])
                                actualResults.append([total])
                        line_count += 1
                    if player_found == False:
                        actualResults.append([0])

                
            return input_data, actualResults
        else:
            return input_data

