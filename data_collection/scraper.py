'''
Daily data collection... scraped game result data from baseball-reference.com and then upcoming games for next period
to create dataframe to feed into model.

**NOTE** While the model is in production right now, there is no 2020 baseball season at the moment due to COVID19.
Thus, the model is currently using a timedelta of 1 year: it is simulating and offering predictions for the 2019
season as if it was being played as the 2020 season. 
'''
from random import choice
import re
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Comment
import requests
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
import time 
import os 
import sys
import json

USER_AGENT = [
    r"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"
]

BASE_URL = 'https://www.baseball-reference.com/boxes/?date={}-{}-{}'

BOX_SCORE = "https://www.baseball-reference.com{}"

OUTPUT_DIR = "./all_data/"

OUTPUT_PATH = "./all_data/current_season.csv"

ELO_LOC = "https://projects.fivethirtyeight.com/mlb-api/mlb_elo_latest.csv"

def prep_dir(output_dir = OUTPUT_DIR):

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)
        
def write_elo(path = ELO_LOC, output_dir = OUTPUT_DIR):

    try:
        
        elo = pd.read_csv(path)

    except Exception as e:

        print("There was an error handling the Elo data: {}".format(e))

    elo.to_csv("{}daily_elo.csv.gz".format(OUTPUT_DIR), compression = "gzip", index = False)

    print("Elo data written to disk")

@dataclass
class BRScraper():
    date: str = None #YYYY-MM-DD format only
    base_url: str = BASE_URL
    box_score: str = BOX_SCORE
    output_path: str = OUTPUT_PATH
    user_agents: list = None
    proxies: dict = None
    scrape_delay: int = 5
    daily_links: list = field(default_factory = list)
    game_soups: list = field(default_factory = list)
    parsed_games: list = field(default_factory = list)
    today_games: list = field(default_factory = list)

    def get_today_games(self):

        self.daily_scrape()

        with open("./all_data/team_map.json", "r+") as f:

            team_map = json.load(f)

        for game_soup in self.game_soups:

            d = {}

            d["date"] = self.date

            teams = [i.get_text() for i in game_soup.find('div', 
            {'class' : 'scorebox'}).findAll('a', {'itemprop' : 'name'})]

            teams = ["".join(team.split(" ")) for team in teams]
            
            teams = [i.replace(".", "") for i in teams]

            d["road_team"], d["home_team"] = team_map[teams[0]], team_map[teams[1]]

            comment_wrappers = game_soup.findAll('div', {'class' : 'section_wrapper setup_commented commented'})
            
            for wrapper in comment_wrappers:

                if wrapper.find('span', {'data-label' : 'Pitching Lines and Info'}):

                    all_pitching = BeautifulSoup(wrapper.find(text = lambda text: isinstance(text, Comment)),
                                                'html.parser')

            for i in range(len(teams)):
    
                tag3 = "all_" + teams[i] + "pitching"
                
                starting_pitching = all_pitching.find('div', {'id' : tag3}).find('tbody').findAll('tr')[0]
                
                starter =  starting_pitching.find('th')['data-append-csv']
                
                if i == 0:

                    d["road_starter"] = starter

                else:

                    d["home_starter"] = starter

            self.today_games.append(d)

        game_frames = pd.DataFrame(self.today_games)

        game_frames.to_csv("./all_data/today_games.csv", index = False)

        print("Today's games have been written to disk")

    def write_to_file(self):
        
        if not self.parsed_games:
            
            raise ValueError("daily_scrape and parse_data methods must be run first")
        
        all_batting = pd.DataFrame([i[0] for i in self.parsed_games])
        
        all_pitching = pd.DataFrame([i[1] for i in self.parsed_games])

        team_map = {
        
        "CincinnatiReds" : "CIN",
        "KansasCityRoyals" : "KCR",
        "LosAngelesDodgers" : "LAD",
        "MiamiMarlins" : "FLA",
        "MilwaukeeBrewers" : "MIL",
        "MinnesotaTwins" : "MIN",
        "NewYorkYankees" : "NYY",
        "OaklandAthletics" : "OAK",
        "PhiladelphiaPhillies" : "PHI",
        "SanDiegoPadres" : "SDP",
        "SeattleMariners" : "SEA",
        "TampaBayRays" : "TBD",
        "TexasRangers" : "TEX",
        "TorontoBlueJays" : "TOR",
        "WashingtonNationals" : "WSN",
        "AtlantaBraves" : "ATL",
        "ClevelandIndians" : "CLE",
        "PittsburghPirates" : "PIT",
        "LosAngelesAngels" : "ANA",
        "BaltimoreOrioles" : "BAL",
        "DetroitTigers" : "DET",
        "NewYorkMets" : "NYM",
        "ArizonaDiamondbacks" : "ARI",
        "ChicagoWhiteSox" : "CHW",
        "ColoradoRockies" : "COL",
        "HoustonAstros" : "HOU",
        "SanFranciscoGiants" : "SFG",
        "StLouisCardinals" : "STL",
        "BostonRedSox" : "BOS",
        "ChicagoCubs" : "CHC"
        
        }

        dfs = [all_batting, all_pitching]

        updated_frames = []

        for df in dfs:
            
            df["date"] = pd.to_datetime(df.date, format = "%Y-%m-%d")
            
            df['year'] = pd.DatetimeIndex(df.date).year
            
            df = df.assign(is_doubleheader = 0, is_tripleheader = 0)
            
            game_counts = df.groupby('home').date.value_counts()

            double_headers = game_counts[game_counts == 2]
            triple_headers = game_counts[game_counts > 2]

            all_double_headers = []
            for j in double_headers.index:
                all_double_headers.append(j)

            all_triple_headers = []
            for k in triple_headers.index:
                all_triple_headers.append(k)

            for index in all_double_headers:
                game_indices = df[(df.home == index[0]) & (df.date == index[1])].index
                if len(game_indices) > 1:
                    df.at[game_indices[1], 'is_doubleheader'] = 1
                else:
                    print(index)

            for index_ in all_triple_headers:
                game_indices_ = df[(df.home == index_[0]) & (df.date == index_[1])].index
                if len(game_indices_) == 3:
                    df.at[game_indices_[1], 'is_doubleheader'] = 1
                    df.at[game_indices_[2], 'is_tripleheader'] = 1
                else:
                    print(index_)
                    
            updated_frames.append(df)

        merge_cols = ["date", "visitor", "home", "is_doubleheader", "is_tripleheader", "year"]

        full_frame = updated_frames[0].merge(updated_frames[1], how = "left", left_on = merge_cols,
                                            right_on = merge_cols)

        full_frame["home_team"] = full_frame.home.map(team_map)

        full_frame["road_team"] = full_frame.visitor.map(team_map)

        full_frame = full_frame.drop(columns = ["home", "visitor"])

        full_frame = full_frame.drop(columns = ["visitorstarter", "homestarter"])

        full_frame = full_frame.rename({"homeretrosheet_id" : "home_starter",
                                    "visitorretrosheet_id" : "road_starter"}, axis = 1)

        all_cols = list(full_frame.columns)

        updated_cols = []

        for col in all_cols:
            
            col = col.replace("visitorstarter", "road_starter_")
            
            col = col.replace("visitorbullpen", "road_relief_")
            
            col = col.replace("homestarter", "home_starter_")
            
            col = col.replace("homebullpen", "home_relief_")
            
            if "starter" not in col or "relief" not in col:
                
                col = col.replace("home", "home_")
                
                col = col.replace("visitor", "road_")
                
            col = col.replace("__", "_")
            
            col = col.replace("SO", "K")
            
            updated_cols.append(col) 

        full_frame.columns = updated_cols

        prefix = ["home_", "road_"]

        for pre in prefix:
            
            full_frame[pre + "TB"] = (full_frame[pre + "HR"] * 4) + (full_frame[pre + "3B"] * 3) +\
            (full_frame[pre + "2B"] * 2) + full_frame[pre + "1B"]

        full_frame = full_frame.rename(columns = {"year" : "season"})

        prefix = ["home_", "road_", "home_starter_", "home_relief_", "road_starter_", "road_relief_"]

        for pre in prefix:
            
            full_frame[pre + "SAC"] = full_frame[pre + "SF"] + full_frame[pre + "SH"]
            

        full_frame = full_frame.drop(columns = ["home_SF", "home_SH", "road_SF", "road_SH",
                                            "home_starter_SF", "home_starter_SH", "home_relief_SF",
                                            "home_relief_SH", "road_starter_SF", "road_starter_SH", "road_relief_SH",
                                            "road_relief_SF"])

        cols = ["home_starter_IP", "road_starter_IP", "home_relief_IP", "road_relief_IP"]

        for col in cols:
            
            full_frame[col] = full_frame[col].astype("str")
            
            for k in range(len(full_frame)):
                
                full_frame.at[k, col] = full_frame.iloc[k][col].replace(".1", ".33")
                
                full_frame.at[k, col] = full_frame.iloc[k][col].replace(".2", ".67")
                
                full_frame.at[k, col] = full_frame.iloc[k][col].replace(".8", ".33")
                
                full_frame.at[k, col] = full_frame.iloc[k][col].replace(".9", ".67")
                
            full_frame[col] = full_frame[col].astype("float32")

        prefix = ["home_starter_", "road_starter_", "home_relief_", "road_relief_"]

        for pre in prefix:
            
            full_frame[pre + "AB"] = full_frame[pre + "H"] + (full_frame[pre + "IP"] * 3).astype("int32")
            
            full_frame[pre + "PA"] = (full_frame[pre + "IP"] * 3).astype("int32") + full_frame[pre + "H"] +\
            full_frame[pre + "BB"] + full_frame[pre + "IBB"] + full_frame[pre + "HBP"] +\
            full_frame[pre + "SAC"]

        prefix = ["home_starter_", "road_starter_", "home_relief_", "road_relief_"]

        for pre in prefix:
            
            full_frame[pre + "1B"] = full_frame[pre + "H"] - full_frame[pre + "2B"] + full_frame[pre + "3B"] +\
            full_frame[pre + "HR"]

        with open("./adv_metric_constants/wOBA_weights.json", "r+") as f:
            
            wOBA_weights = json.load(f)

        wOBA = pd.DataFrame(wOBA_weights)

        change_cols = ["wOBA", "wOBAScale", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR",
                    "runSB", "runCS", "R/PA", "R/W", "cFIP"]

        wOBA["Season"] = wOBA.Season.astype("int64")

        for col in change_cols:
            
            wOBA[col] = wOBA[col].astype("float32")
            
        wOBA = wOBA.rename(columns = {"Season" : "season"})

        full_frame = full_frame.merge(wOBA, how = "left", left_on = ["season"],
                                right_on = ["season"])

        teams = requests.get("https://www.retrosheet.org/TEAMABR.TXT").content

        team_soup = BeautifulSoup(teams, "html.parser").get_text().split("\n")

        leagues = {}

        pattern = r'([\w]+)'

        for row in team_soup:
                
            vals = re.findall(pattern, row)
            
            try:
            
                leagues[vals[0]] = vals[1]
                
            except:
                
                continue

        with open("./adv_metric_constants/modern_rc.json", "r+") as f:
            retrosheet_codes = json.load(f)
            
        retrosheet_codes.update({"MIA" : "FLA"})

        pop_list = []

        for key in leagues:
            
            if key not in retrosheet_codes:
                
                pop_list.append(key)

        for key in pop_list:
            
            leagues.pop(key)

        leagues["MIA"] = "NL"

        df = pd.DataFrame({"team_code" : list(leagues.keys()), "league" : list(leagues.values())})

        df["elo_code"] = df.team_code.map(retrosheet_codes)

        elo_leagues = {}

        for key in retrosheet_codes:
            
            elo_leagues[retrosheet_codes[key]] = leagues[key]

        elo_leagues["HOU"] = "AL"

        full_frame["home_league"] = full_frame.home_team.map(elo_leagues)

        full_frame["road_league"] = full_frame.road_team.map(elo_leagues)

        al_league_wRC = pd.read_csv("./adv_metric_constants/league_wRC_AL.csv")

        nl_league_wRC = pd.read_csv("./adv_metric_constants/league_wRC_NL.csv")

        al_league_wRC = al_league_wRC[["Season", "PA", "wRC"]].rename(columns = {
            "Season" : "season",
            "PA" : "league_PA",
            "wRC" : "league_wRC"
        })

        nl_league_wRC = nl_league_wRC[["Season", "PA", "wRC"]].rename(columns = {
            "Season" : "season",
            "PA" : "league_PA",
            "wRC" : "league_wRC"
        })

        al_league_wRC = al_league_wRC.assign(league = "AL")

        nl_league_wRC = nl_league_wRC.assign(league = "NL")

        league_wRC = pd.concat([al_league_wRC, nl_league_wRC], axis = 0).sort_values(by = ["season"]).reset_index(drop = True)

        full_frame = full_frame.merge(league_wRC, how = "left", left_on = ["season", "home_league"],
                            right_on = ["season", "league"])

        full_frame = full_frame.drop(columns = ["league"]).rename(columns = {
            "league_PA" : "home_league_PA",
            "league_wRC" : "home_league_wRC"
        }
        )

        full_frame = full_frame.merge(league_wRC, how = "left", left_on = ["season", "road_league"],
                            right_on = ["season", "league"])

        full_frame = full_frame.drop(columns = ["league"]).rename(columns = {
            "league_PA" : "road_league_PA",
            "league_wRC" : "road_league_wRC"
        })

        all_stadiums = pd.read_csv("./adv_metric_constants/all_stadiums_w_park_ids.csv", index_col = 0)

        park_factors = all_stadiums[["team_code", "year", "batting_park_factor"]]

        full_frame = full_frame.merge(park_factors, how = "left", 
                                left_on = ["home_team", "season"],
                                right_on = ["team_code", "year"])

        full_frame = full_frame.drop(columns = ["team_code", "year"])

        full_frame = full_frame.rename(columns = {"batting_park_factor" : "home_batting_park_factor"})

        full_frame.home_batting_park_factor = full_frame.home_batting_park_factor / 100.
        
        if os.path.isfile(self.output_path) or os.path.islink(self.output_path):
            
            with open(self.output_path, "a") as f:
                
                full_frame.to_csv(f, header = False, index = False)
                
        else:
            
            full_frame.to_csv(self.output_path, index = False)
            
        print("All data written to file for data {}".format(self.date))
        
    def parse_data(self):

        for game_soup in self.game_soups:
    
            batting = {}

            pitching = {}

            batting["date"] = self.date

            pitching["date"] = self.date

            teams = [i.get_text() for i in game_soup.find('div', 
            {'class' : 'scorebox'}).findAll('a', {'itemprop' : 'name'})]

            teams = ["".join(team.split(" ")) for team in teams]
            
            teams = [i.replace(".", "") for i in teams]
            
            batting["visitor"], pitching["visitor"] = teams[0], teams[0]
            
            batting["home"], pitching["home"] = teams[1], teams[1]
            
            comment_wrappers = game_soup.findAll('div', {'class' : 'section_wrapper setup_commented commented'})
            
            for wrapper in comment_wrappers:
                    
                if wrapper.find('span', {'data-label' : 'Pitching Lines and Info'}):

                    all_pitching = BeautifulSoup(wrapper.find(text = lambda text: isinstance(text, Comment)),
                                                'html.parser')
                    
            for i in range(len(teams)):

                if i == 0:

                    prefix = "visitor"

                    alt_prefix = "home"

                else:

                    prefix = "home"

                    alt_prefix = "visitor"

                tag1 = "all_" + teams[i] + "batting"

                tag2 = "tfooter_" + teams[i] + "batting"
                
                tag3 = "all_" + teams[i] + "pitching"

                team_batting = BeautifulSoup(game_soup.find('div', {'id' : tag1}).\
                            find(text = lambda text: isinstance(text, Comment)), 'html.parser')

                to_collect1 = ["AB", "H", "PA", "BB", "R"]

                for stat in to_collect1:

                    batting[prefix + stat] = int(team_batting.find("tfoot").\
                    find('td', {'data-stat' : stat}).get_text().strip())

                batting_extras = team_batting.find("div", {"id" : tag2})

                to_collect2 = ["HR", "2B", "3B", "SF", "SH", "IBB", "HBP"]

                for i in to_collect2:

                    search_cat = i + prefix

                    category = batting_extras.find("div", {"id" : search_cat})

                    if category:

                        category = category.get_text().split(i + ": ")[1]
                        
                        if i not in ["IBB", "HBP"]:

                            n, p = self.extract_values(category)
                            
                        else:
                            
                            n, p = self.extract_values(category, True)

                        batting[prefix + i] = n

                        pitching[alt_prefix + i] = p

                    else:

                        batting[prefix + i] = 0
                        
                        pitching[alt_prefix + i] = 0
                        
                batting[prefix + "1B"] = batting[prefix + "H"] - batting[prefix + "HR"] - batting[prefix + "3B"] -\
                batting[prefix + "2B"]
                
                starting_pitching = all_pitching.find('div', {'id' : tag3}).find('tbody').findAll('tr')[0]
                
                team_totals = all_pitching.find('div', {'id' : tag3}).find('tfoot')
                
                starting_pitcher_retro_id = starting_pitching.find('th')['data-append-csv']
                
                starting_pitcher_name = starting_pitching.find('th', {'data-stat' : 'player'}).\
                get_text().split(', ')[0]
                
                pitching[prefix + "starter"] = starting_pitcher_name
                
                pitching[prefix + "retrosheet_id"] = starting_pitcher_retro_id
                
                to_collect3 = ['IP', 'H', 'ER', 'BB', 'SO']
                
                for stat in to_collect3:
                    
                    if stat == 'H':
                        
                        hits_surrendered = float(starting_pitching.find('td', {'data-stat' : stat}).get_text().strip())
                        
                        pitching[prefix + 'starter_H'] = hits_surrendered
                        
                        pitching[prefix + 'bullpen_H'] = float(
                            team_totals.find('td', {'data-stat' : stat}).get_text().strip()) - \
                            hits_surrendered
                        
                    else:
                        
                        pitching[prefix + "starter" + stat] = float(
                            starting_pitching.find('td', {'data-stat' : stat}).get_text().strip())
                        
                        pitching[prefix + "bullpen" + stat] = float(
                            team_totals.find('td', {'data-stat' : stat}).get_text().strip()) - \
                        pitching[prefix + "starter" + stat]
                        
            for prefix in ["visitor", "home"]:
                
                check_keys = [(prefix + "HR", "HR"), (prefix + "2B", "2B")
                    , (prefix + "3B", "3B"), (prefix + "SF", "SF"), 
                    (prefix + "SH", "SH"), (prefix + "IBB", "IBB"), (prefix + "HBP", "HBP")]

                for key, label in check_keys:

                    starter = pitching[prefix + "starter"]

                    f = pitching[key]

                    if type(f) == int:

                        pitching[prefix + "starter" + label] = 0

                        pitching[prefix + "bullpen" + label] = 0

                        pitching.pop(key)

                    else:

                        if starter in f:

                            pitching[prefix + "starter" + label] = f[starter]

                            f.pop(starter)

                            pitching[prefix + "bullpen" + label] = sum(f.values())

                            pitching.pop(key)

                        else:

                            pitching[prefix + "starter" + label] = 0

                            pitching[prefix + "bullpen" + label] = sum(f.values())

                            pitching.pop(key)

            self.parsed_games.append([batting, pitching])

        print("All game data parsed for {}".format(self.date))

    def daily_scrape(self):

        home_url = self.build_query(self.base_url, self.date)

        agent = self.random_agent()
            
        try:
            response = requests.get(home_url, headers = {

            "User-Agent" : agent

        },

            proxies = self.proxies)
        
            response.raise_for_status()

        except HTTPError:

            raise

        except RequestException:

            raise

        else:

            home_soup = BeautifulSoup(response.content, "html.parser")

        box_scores = home_soup.find('div', {'class' : 'game_summaries'}).findAll('table', {'class' : 'teams'})
        
        for box in box_scores:

            a_tags = box.findAll('a')

            for a in a_tags:

                if '/boxes/' in a['href']:

                    self.daily_links.append(a['href'])

                    break

        for link in self.daily_links:

            box_url = self.box_score.format(link)

            agent = self.random_agent()

            try:
                response = requests.get(box_url, headers = {

                "User-Agent" : agent

            },

                proxies = self.proxies)
            
                response.raise_for_status()

            except HTTPError:

                raise

            except RequestException:

                raise

            else:

                game_soup = BeautifulSoup(response.content, "html.parser")

            self.game_soups.append(game_soup)

            time.sleep(self.scrape_delay)

        print("Scraping complete for {}...".format(self.date))

    def random_agent(self):
        
        if self.user_agents and isinstance(self.user_agents, list):
        
            return(choice(self.user_agents))

        return(choice(USER_AGENT))
    
    @staticmethod
    def build_query(base, date = None):
        ''' 
        Factory method- build daily URL for baseball-reference.com box score homepage for previous day.
        NOTE: To alter data collection frequency, simply change timedelta value
        '''

        if not date:

            date = datetime.now() - timedelta(days = 1)

            return(base.format(date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')))

        else:
            
            try:
                
                date = datetime.strptime(date, "%Y-%m-%d")
                
            except Exception:
                
                raise ValueError("Please pass date format of '%Y-%m-%d'")
                
            return(base.format(date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')))

    @staticmethod
    def extract_values(s, BB_category = False):
    
        if not BB_category:
        
            n_events = 0

            pitchers = {}

            l = s.split("); ")

            solo_pitcher_match = r"off\s[\w\s\.]*"

            multi_pitcher_match = r"[\d]*\soff\s[\w\s\.]*"

            for value in l:

                if value.split(" (")[0][-1] not in "123456879":

                    n_events += 1

                    pitcher = re.findall(solo_pitcher_match, value)[0].split("off ")[1]

                    if pitcher not in pitchers:
                        
                        pitchers[pitcher] = 1 
                        
                    else:
                        
                        pitchers[pitcher] += 1

                else:

                    n_events += int(value.split(" (")[0][-1])

                    pitcher = re.findall(multi_pitcher_match, value)

                    if len(pitcher) >= 1:

                        for event in pitcher:

                            n_surrendered = int(event[0])

                            name = event.split("off ")[1]
                            
                            if name not in pitchers:

                                pitchers[name] = n_surrendered
                                
                            else:
                                
                                pitchers[name] += n_surrendered

            return(n_events, pitchers)
        
        else:
        
            n_events = 0

            pitchers = {}

            l = s.split("); ")

            solo_pitcher_match = r"by\s[\w\s\.]*"

            multi_pitcher_match = r"[\d]*\sby\s[\w\s\.]*"

            for value in l:

                if value.split(" (")[0][-1] not in "123456879":

                    n_events += 1

                    pitcher = re.findall(solo_pitcher_match, value)[0].split("by ")[1]

                    if pitcher not in pitchers:
                        
                        pitchers[pitcher] = 1 
                        
                    else:
                        
                        pitchers[pitcher] += 1

                else:

                    n_events += int(value.split(" (")[0][-1])

                    pitcher = re.findall(multi_pitcher_match, value)

                    if len(pitcher) >= 1:

                        for event in pitcher:

                            n_surrendered = int(event[0])

                            name = event.split("by ")[1]
                            
                            if name not in pitchers:

                                pitchers[name] = n_surrendered
                                
                            else:
                                
                                pitchers[name] += n_surrendered

            return(n_events, pitchers)

if __name__ == "__main__":
    '''
    Currently, running with a timedelta of 1 year. When current season starts, just remove this piece
    '''
    today = datetime.strptime(datetime.strftime(datetime.now(), "%Y-%m-%d"), "%Y-%m-%d")

    past_day = datetime.strftime(today - timedelta(days = 367), "%Y-%m-%d")

    future_day = datetime.strftime(today - timedelta(days = 366), "%Y-%m-%d")

    print("Beginning collection of games played on {}".format(past_day))
    
    BRS = BRScraper(past_day)

    BRS.daily_scrape()

    BRS.parse_data()

    BRS.write_to_file()

    print("Beginning collection of games being played today, on {}".format(future_day))

    BRS = BRScraper(future_day)

    BRS.get_today_games()
    
