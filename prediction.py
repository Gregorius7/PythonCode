from flask import Flask
import pandas as pd
import numpy as np
import math
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient, errors
import pickle

app = Flask(__name__)
@app.route('/')
def Index():
    selected = GetDocument()
    my_id = selected['id']
    fileName = selected['attributes']['file']
    time = selected['attributes']['time']
    player = selected['attributes']['name']

    # Delete ->
    # client.exampleDB.dataframeTest.deleteOne( { id: my_id} )

    df = pd.read_parquet(fileName)
    balls = df.loc[(df['player_name'] == 'ball')].copy()
    players = df.loc[(df['player_name'] != 'ball')].copy()
    
    pitch_cells_x = 20
    pitch_cells_y = 12
    ball_posession_by_sec = {}
    
    return str("Value is: " + str( round(PredictResult(players, balls, df['half'].values[0], time, player), 2)))

def CreateDataframe_ByDF(team_data, ball_data):
    # Az előzetes adatok
    df_tmplt = {'half': [],
                'time': [],
                'team_name': [],
                'player_name': [],
                'pos_x': [],
                'pos_y': [],
                'speed': [],
                'mov_dir': [],
                'dir': [],
                'next_pos_x': [],'next_pos_y': [],'next_cell_id': [],
                'previous_pos_x': [],'previous_pos_y': [],'previous_cell_id': [],
                'five_sec_previous_pos_x': [],'five_sec_previous_pos_y': [],'five_sec_previous_cell_id': []
               }
    
    # Létrehozunk egy nagy dataframet az összes játékossal és labdával
    prepared_df = pd.DataFrame(df_tmplt, columns = ['half', 'time', 'team_name', 'player_name', 'pos_x', 'pos_y', 'speed', 'mov_dir', 'dir', 'next_pos_x', 'next_pos_y', 'next_cell_id', 'previous_pos_x','previous_pos_y','previous_cell_id',
                                           'five_sec_previous_pos_x','five_sec_previous_pos_y','five_sec_previous_cell_id'])
    
    prepared_df = prepared_df.append(team_data)
    prepared_df = prepared_df.append(ball_data)
    
    prepared_df = prepared_df.sort_values(by=['half','time', 'player_name'])
    #prepared_df = prepared_df.set_index(['half', 'time', 'player_name'])

    assigned_cols = ['time', 'half', 'team_name', 'player_name', 'pos_x', 'pos_y', 'speed']
    for col in prepared_df.columns:
        if col not in assigned_cols:
            prepared_df[col] = 999

    return prepared_df

# A hiányzó labdák kitöltése a legutolsó, és az elkövetkezendő felvett labda helyzetének ismerete alapján
def FillMissing_Balls(ball_data):
    ball_data = ball_data.sort_values(by=['half', 'time'])
    final_ball_data = None

    # Mindkét félidőre megnézzük a legelső és legutolsó rögzített időpontot
    # E között a két idő között minden hiányzó időpontra létrehozunk egy nan sort
    # A nan sorokban az ismert előzetes és elkövetkező labda pozíciók alapján interpolate függvény kipótolja a pozíciókat
    for half in ball_data['half'].unique():
        checked_half = ball_data.loc[ball_data['half'] == half]

        time_min = checked_half['time'].unique().min()
        time_max = checked_half['time'].unique().max()
        checked_half = checked_half.set_index("time") # Time-ot használjuk indexnek
        new_index = pd.Index(np.arange(time_min,time_max,1), name="time") # 1-esével lépkedve ahol hiányzik index, felveszünk új sort
        checked_half = checked_half.reindex(new_index)
        checked_half = checked_half.reset_index() # Visszaállítjuk az idő oszlopot

        # Az új nan sorokban felvesszük az alap értékeket
        checked_half.loc[checked_half['half'].isnull(), ['team_name', 'player_name', 'half']] = ["estimated", "ball", half]
        
        # Ugyan ezeknek a soroknak a pozícióit interpolate-el kipótoljuk
        checked_half = checked_half.interpolate(method='linear', limit_direction='forward', axis=0)
        
        if final_ball_data is None:
            final_ball_data = checked_half.copy()
        else:
            final_ball_data = final_ball_data.append(checked_half)

    return final_ball_data

# Kiszámolja a jelenlegi, és tényleges mozgás irányát, illetve előző/jelenlegi/jövőbeli cellákat, pozíciókat.
def CalculateDirectionsPositions(df):
    
    # Oszlopok létrehozása
    df['mov_dir'] = 999
    df['previous_pos_x'] = 999
    df['previous_pos_y'] = 999
    df['p_previous_pos_x'] = 999
    df['p_previous_pos_y'] = 999
    
    # Sorbarendezzük minden játékosnak az összes rögzített idejét, mindkét félidőre
    df = df.sort_values(by=['half','player_name', 'time'])
    for half in [1,2]:
        for player in df.loc[df["half"] == half]["player_name"].unique():
            mask = ((df["half"] == half) & (df["player_name"] == player))

            # Hogy megkapjuk az előző pozíciót, minden játékos adott időpontbeli pozícióját a következő időponthoz shifteljük
            df.loc[mask, 'previous_pos_x'] = df.loc[mask]['pos_x'].shift()
            df.loc[mask, 'previous_pos_y'] = df.loc[mask]['pos_y'].shift()
            df.loc[mask, 'p_previous_pos_x'] = df.loc[mask]['pos_x'].shift(2)
            df.loc[mask, 'p_previous_pos_y'] = df.loc[mask]['pos_y'].shift(2)

    df = df.dropna()
    
    # Miután ezek megvannak, kitudjuk számolni a mozgás irányokat majd cellákat
    df['mov_dir'] = df.apply(lambda row: math.degrees(math.atan2(row['pos_y'] - row['previous_pos_y'], row['pos_x'] - row['previous_pos_x'])), axis=1)

    return df

# A labdához legközelebbi játékos és annak csapatának nevének lekérdezése
def BallAt(df_time):
    # Adott időpont játékosait és labdáját vizsgáljuk
    
    players = df_time.loc[df_time.index[0][0], df_time.index[0][1], set(df_time.index.levels[2]) - set(["ball"])].copy()
    ball = None
    if (df_time.index[0][0], df_time.index[0][1], "ball") in df_time.index:
        ball = df_time.loc[df_time.index[0][0], df_time.index[0][1], "ball"].copy()
    else:
        return ("None", "None")
    
    if (players is None) or (ball.shape[0] <= 0) or (players.shape[0] <= 0):
        return ("None", "None")

    if not (ball.iloc[0] == "fix" or ball.iloc[0] == "estimated"):
        ball = ball.iloc[0]

    # Ezekre a játékosokra meghatározzuk a labdátol vett távolságot
    players['dist_from_ball'] = 999
    for row in players.itertuples():
        players.at[row.Index, 'dist_from_ball'] == math.sqrt(  pow(ball['pos_x'] - row.pos_x, 2) + pow(ball['pos_y'] - row.pos_y,2)  )
    
    # Sorbarendezzük eszerint a táv szerint
    players = players.sort_values(by=['dist_from_ball'])
    
    closest_player = players.iloc[0].name[2]
    closest_team = players.iloc[0]['team_name']
    return (closest_player, closest_team, ball['pos_x'], ball['pos_y'])

# Azonosítóval rendelkező cella lekérdezése pozíció alapján
def GetCellID(pos):
    player_pitch_x = math.floor((pos[0] + 52) / (104/pitch_cells_x))
    player_pitch_y = math.floor((pos[1]) / (68/pitch_cells_y))
    if player_pitch_y >= pitch_cells_y:
        player_pitch_y = pitch_cells_y-1
    if player_pitch_x >= pitch_cells_x:
        player_pitch_y = pitch_cells_x-1
    return ( ((player_pitch_y*pitch_cells_x) + 1) + pitch_cells_x)

# Egy tipp mely csapat van a pálya bal oldalán az adott félidőben.
def GetTeamLeft_Positions(df, half):
    team_names = df['team_name'].unique()
    
    # A két csapat első néhány sorait vizsgáljuk
    df_team_1 = df.loc[df['team_name'] == team_names[0]][:50]
    df_team_2 = df.loc[df['team_name'] == team_names[1]][:50]
    
    # Megszámoljuk melyik csapatnak van több játékosa a pálya bal oldalán
    team_1_x_less_than_zero = df_team_1[ df_team_1['pos_x'] < 0 ].count()['pos_x']
    team_2_x_less_than_zero = df_team_2[ df_team_2['pos_x'] < 0 ].count()['pos_x']

    if team_1_x_less_than_zero > team_2_x_less_than_zero:
        return team_names[0]
    else:
        return team_names[1]
    
# Vissza adja éppen támad-e a vizsgált csapat az adott időpontban
def IsTeamAttacking_Positions(df_time, team, team_left):
    df_only_team = df_time.loc[(df_time["team_name"] == team)]
    
    # Megnézzük mely csapatból haladnak többen a saját térféllel ellentétes irányba
    going_left =  df_only_team[ ((df_only_team['mov_dir'] >= 110.0)  & (df_only_team['mov_dir'] <=  180.0)) ].count()['mov_dir']
    going_left += df_only_team[ ((df_only_team['mov_dir'] <= -110.0) & (df_only_team['mov_dir'] >= -180.0)) ].count()['mov_dir']                                         
                                 
    going_right = df_only_team[ ((df_only_team['mov_dir'] <= 70.0)  & (df_only_team['mov_dir'] >= -70.0))].count()['mov_dir']
    
    # Ha a bal térfél a csapaté, és jobbra mozognak, feltételezhetően támadnak.
    if (team_left == team) and going_right > going_left:
        return True
    # Ha a bal térfél nem a csapaté, és balra mozognak, feltételezhetően támadnak.
    if not (team_left == team) and going_left > going_right:
        return True
    
    # Máskülönben nem támadnak.
    return False

def GetTeammates(df_time, team, player, pos_x, pos_y):
    df_used = df_time.filter(items=['player_name', 'team_name', 'pos_x', 'pos_y']).copy()
    # Megnézzük kik az azonos csapatbeli játékosok az adott időpontban
    other_players = df_used.loc[df_time.index[0][0], df_time.index[0][1], (set(df_time.index.levels[2]) - set([player])), df_used['team_name'] == team].copy()
    other_players = other_players.filter(items=['player_name', 'pos_x', 'pos_y']).copy()

    # Kiszámoljuk az összes másik játékos távolságát a vizsgált játékostól, majd sorbarendezzük eszerint
    for row in other_players.itertuples():
        other_players.at[row.Index, 'dist_from_player'] = math.sqrt(  pow(pos_x - row.pos_x, 2) + pow(pos_y - row.pos_y,2))

    if other_players.shape[0] <= 0:
        return None
    
    other_players = other_players.sort_values(by=['dist_from_player'])
    
    return other_players

def GetOpponents(df_time, team, player, pos_x, pos_y):
    df_used = df_time.filter(items=['player_name', 'team_name', 'pos_x', 'pos_y']).copy()
    
    # Megnézzük kik a másik csapat játékosai az adott időpontban
    other_players = df_used.loc[(df_used['team_name'] != team)].copy()
    other_players = other_players.filter(items=['player_name', 'pos_x', 'pos_y']).copy()
    
    # Kiszámoljuk az összes másik játékos távolságát a vizsgált játékostól, majd sorbarendezzük eszerint
    for row in other_players.itertuples():
        other_players.at[row.Index, 'dist_from_player'] = math.sqrt(  pow(pos_x - row.pos_x, 2) + pow(pos_y - row.pos_y,2))

    if other_players.shape[0] <= 0:
        return None
    
    other_players = other_players.sort_values(by=['dist_from_player'])

    return other_players

# A dataframe csak azon sorait tarja meg, amely sorok meccs időpontjaiban labda is van
def TimesWithBall(df):
    df_final = None
    for h in df.index.levels[0]:
        half_h_ball_times = df.loc[h].index.levels[0]
        first_ball_time = half_h_ball_times.min()
        last_ball_time =  half_h_ball_times.max()
        df_h = df.loc[h, first_ball_time:last_ball_time, :].copy()
        
        if df_final is None:
            df_final = df_h
        else:
            df_final.append(df_h)
    
    return df_final

# Itertuples-ként végigmegyünk és minden sorra kiszámoljuk és felveszzük az új oszlopokat
def Fill_DF_Ver2(df):
    df = df.sort_values(by=['half','time', 'player_name'])
    df = df.set_index(['half', 'time', 'player_name'])
    
    team_names = df[(df["team_name"] != "estimated") & (df["team_name"] != "fix")]['team_name'].unique()
    checked_team = team_names[0] # Vizsgált csapat, mindig erre határozzuk meg támad-e
    team_left_by_half = {} # Melyik csapat van bal oldalt, mely félidőbe
    global ball_posession_by_sec # Labdához legközelebbi játékos és csapata, illetve labda pozíció a két félidő minden másodpercére
    checked_team_attack_by_sec = {} # A vizsgált csapat támad-e az adott időpontban
    avg_for_players_by_half = {} # A játékosok félidőnkénti átlagos pozíciója
    for h in ["1.0", "2.0"]:
        avg_for_players_by_half[h] = {}
        ball_posession_by_sec[h] = {}
        checked_team_attack_by_sec[h] = {}

    # Kitörlök minden olyan időpontot, ahol nincs labda
    df = TimesWithBall(df)
    df_filtered = df.filter(items=['half', 'time', 'player_name', 'team_name', 'pos_x', 'pos_y', 'mov_dir', 'speed']).copy()
    timer = 0
    for row in df_filtered.itertuples():
        # Csak a hátralévő idő mutatása
        timer += 1
        if timer % int(int((df_filtered.shape[0]) / 25)) == 1:
            pass
            #print( str( round( (float(timer)/ (int(df_filtered.shape[0])*1.0) )*100.0,2)) + "%")
        #
        
        if (row.speed < 1.5) or (row.Index[2] == 'ball'):
            continue
        
        # Csak a vizsgált sor idejével azonos sorok
        
        df_time = df_filtered.loc[ [row.Index[0], row.Index[1]] ].copy()  #<- Nagy futási különbség!!
        # df_time = df_filtered.loc[row.Index[0], row.Index[1], :].copy()
        
        if str(row.Index[0]) not in team_left_by_half:
            team_left_by_half[str(row.Index[0])] = GetTeamLeft_Positions(df_filtered, row.Index[0])
            
        if str(row.Index[1]) not in ball_posession_by_sec[str(row.Index[0])]:
            ball_posession_by_sec[str(row.Index[0])][str(row.Index[1])] = BallAt(df_time)
            checked_team_attack_by_sec[str(row.Index[0])][str(row.Index[1])] = IsTeamAttacking_Positions(df_time, checked_team, team_left_by_half[str(row.Index[0])])
        
        # Távolság szerint sorbarendezett ellenfelek / csapattársak
        opponents = GetOpponents(df_time, row.team_name, row.Index[2], row.pos_x, row.pos_y)
        teammates = GetTeammates(df_time, row.team_name, row.Index[2], row.pos_x, row.pos_y)

        if (opponents is None) or (teammates is None):
            continue
            
        df.at[row.Index, 'is_attacking'] = ( ((row.team_name == checked_team) & (checked_team_attack_by_sec[str(row.Index[0])][str(row.Index[1])])) or ((row.team_name != checked_team) & (not checked_team_attack_by_sec[str(row.Index[0])][str(row.Index[1])])) )
        df.at[row.Index, 'player_has_ball'] = ( row.Index[2] == ball_posession_by_sec[str(row.Index[0])][str(row.Index[1])][0] )
        df.at[row.Index, 'ball_poss'] = ( row.team_name == ball_posession_by_sec[str(row.Index[0])][str(row.Index[1])][1] )

        if ball_posession_by_sec[str(row.Index[0])][str(row.Index[1])][0] == "None":
            continue

        ball_pos = (ball_posession_by_sec[str(row.Index[0])][str(row.Index[1])][2], ball_posession_by_sec[str(row.Index[0])][str(row.Index[1])][3])
        
        if (row.Index[0], row.Index[1], 'ball') in df.index:
            
            df.at[row.Index, 'ball_pos_x'] = ball_pos[0]
            df.at[row.Index, 'ball_pos_y'] = ball_pos[1]

            df.at[row.Index, 'ball'] = Fix_Direction(math.degrees(math.atan2(ball_pos[1] - row.pos_y,ball_pos[0] - row.pos_x)) - row.mov_dir)
            df.at[row.Index, 'ball_dist'] = math.sqrt(  pow(ball_pos[0] - row.pos_x, 2) + pow(ball_pos[1] - row.pos_y,2))

            # Legközelebbi X. játékos adatai
            num_list = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']
            for i in num_list:
                opponent = opponents.iloc[num_list.index(i)]
                teammate = teammates.iloc[num_list.index(i)]
                
                df.at[row.Index, i + '_opponent_pos_x'] = opponent['pos_x']
                df.at[row.Index, i + '_opponent_pos_y'] = opponent['pos_y']
                df.at[row.Index, i + '_opponent_name'] = opponent.name[2]

                df.at[row.Index, i + '_teammate_pos_x'] = teammate['pos_x']
                df.at[row.Index, i + '_teammate_pos_y'] = teammate['pos_y']
                df.at[row.Index, i + '_teammate_name'] = teammate.name[2]
            
                df.at[row.Index, i + '_opponent'] = Fix_Direction(math.degrees(math.atan2(opponent['pos_y'] - row.pos_y, opponent['pos_x'] - row.pos_x)) - row.mov_dir)
                df.at[row.Index, i + '_opponent_dist'] = math.sqrt(  pow(opponent['pos_x'] - row.pos_x, 2) + pow(opponent['pos_y'] - row.pos_y,2))
                df.at[row.Index, i + '_teammate'] = Fix_Direction(math.degrees(math.atan2(teammate['pos_y'] - row.pos_y, teammate['pos_x'] - row.pos_x)) - row.mov_dir)
                df.at[row.Index, i + '_teammate_dist'] = math.sqrt(  pow(teammate['pos_x'] - row.pos_x, 2) + pow(teammate['pos_y'] - row.pos_y,2))

        if row.Index[2] not in avg_for_players_by_half[str(row.Index[0])]:
            avg_x = df_filtered.loc[row.Index[0], :, row.Index[2]]['pos_x'].mean()
            avg_y = df_filtered.loc[row.Index[0], :, row.Index[2]]['pos_y'].mean()
            avg_cell_id = int(GetCellID( (avg_x, avg_y) ))
            avg_pos_dir = Fix_Direction(math.degrees(math.atan2(avg_y - row.pos_y, avg_x - row.pos_x)) - row.mov_dir)

            avg_for_players_by_half[str(row.Index[0])][row.Index[2]] = (avg_x, avg_y, avg_cell_id, avg_pos_dir)
    
        df.at[row.Index, 'avg_pos_x'] = avg_for_players_by_half[str(row.Index[0])][row.Index[2]][0]
        df.at[row.Index, 'avg_pos_y'] = avg_for_players_by_half[str(row.Index[0])][row.Index[2]][1]
        df.at[row.Index, 'avg_cell_id'] = avg_for_players_by_half[str(row.Index[0])][row.Index[2]][2]
        df.at[row.Index, 'avg_pos'] = avg_for_players_by_half[str(row.Index[0])][row.Index[2]][3]
    
    df = df.round(1)
    
    return df

# Relatív irányok javítása 
# (Ha az alap irányhoz képest 181 fokra van a vizsgált irány, akkor helyesebb, ha úgy tekintjük -179 fok-ra van)
def Fix_Direction(direction):
    if direction > 180.0:
        direction -= 360.0
    if direction < -180.0:
        direction += 360.0
    return direction

# A relatív irányok (E felett lévő algoritmus) visszaállítása
def Reset_Relative_Direction(mov_dir, angle):
    return Fix_Direction(angle + mov_dir)

# Szűrés
def Drop_Errors(df_to_fix):
    df_to_fix.dropna(inplace=True)
    df_to_fix.drop(df_to_fix.loc[df_to_fix['mov_dir']> 180.0].index, inplace=True)
    df_to_fix.drop(df_to_fix.loc[df_to_fix['mov_dir']< -180.0].index, inplace=True)
    df_to_fix.drop(df_to_fix.loc[df_to_fix['dir']> 180.0].index, inplace=True)
    df_to_fix.drop(df_to_fix.loc[df_to_fix['dir']< -180.0].index, inplace=True)
    df_to_fix.drop(df_to_fix.loc[df_to_fix['ball']== None].index, inplace=True)
    df_to_fix.drop(df_to_fix.loc[df_to_fix['team_name']== "fix"].index, inplace=True)
    df_to_fix.drop(df_to_fix.loc[df_to_fix['team_name']== "estimated"].index, inplace=True)
    df_to_fix.drop(df_to_fix.loc[df_to_fix['speed'] < 1.5].index, inplace=True)
    
    # Amennyiben van olyan sor, ahol mov_dir nem az 1 másodpercel későbbi pozícióra lett számolva, töröljük
    everyTime_h1 = df_to_fix.loc[1].index.levels[0]
    everyTime_h2 = df_to_fix.loc[2].index.levels[0]
    for row in df_to_fix.itertuples():
        if row.Index[0] == 1 and ((row.Index[1]-1) not in everyTime_h1) or ((row.Index[1]+1) not in everyTime_h1):
            df_to_fix.loc[row.Index, 'mov_dir'] = 999
        if row.Index[0] == 2 and ((row.Index[1]-1) not in everyTime_h2) or ((row.Index[1]+1) not in everyTime_h2):
            df_to_fix.loc[row.Index, 'mov_dir'] = 999
            
    df_to_fix.drop(df_to_fix.loc[df_to_fix['mov_dir']> 180.0].index, inplace=True)
    #df_to_fix.drop(df_to_fix.loc[df_to_fix['ball_dist'] >= 12.0].index, inplace=True)

    return df_to_fix

def GetDocument():
    DOMAIN = '172.104.251.132'
    PORT = 27017

    try:
        client = MongoClient(
            host = [ str(DOMAIN) + ":" + str(PORT) ],
            serverSelectionTimeoutMS = 3000, # 3 second timeout
            username = "admin",
            password = "buksi77",
        )

        #print ("server version:", client.server_info()["version"])

        database_names = client.list_database_names()

    except errors.ServerSelectionTimeoutError as err:
        client = None
        database_names = []

        #print ("pymongo ERROR:", err)

    example = client.exampleDB.dataframeTest
    return example.find_one()

def PredictResult(df_team, df_ball, half, time, player):
    pr_initial = CreateDataframe_ByDF(df_team, df_ball).copy()
    pr_calc = CalculateDirectionsPositions(pr_initial).copy()
    pr_final = Fill_DF_Ver2(pr_calc).copy()
    pr_record = pr_final.loc[half, time, player].copy()
    
    features = ['pos_x', 'pos_y', 'mov_dir', 'ball','avg_pos', 'speed',
           'first_opponent','second_opponent','third_opponent','fourth_opponent','fifth_opponent','sixth_opponent',
           'first_teammate','second_teammate','third_teammate','fourth_teammate','fifth_teammate','sixth_teammate']
    
   
    x_pred = pr_record[features].copy()
    x_test = []
    x_test.append(x_pred)
    
    clf = None
    with open('my_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    y_pred = clf.predict(x_test)
    
    return y_pred[0]

app.run(host="0.0.0.0", port=int("80"), debug = True)