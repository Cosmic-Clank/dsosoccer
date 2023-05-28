import numpy as np
import cv2
import pyzed.sl as sl
from ColorModule import ColorFinder
import math


class Team:
    def __init__(self, color, team_color):
        self.color = color
        self.team_color = team_color
        self.score = 0
        self.players = {}  # {player_obj_id: 2d coordinates}

    def get_color(self):
        return self.color

    def get_score(self):
        return self.score
    
    def add_score(self, score):
        self.score += score
        
    def goal(self):
        self.score += 1

    def get_players(self):
        return self.player

    def add_player(self, player_obj, window_width, window_height):
        self.players[player_obj.id] = generate_2d_obj_position(
            player_obj, window_width, window_height)

    def clear_players(self):
        self.players.clear()

    def update_football_map(self, ground):
        for player_id in self.players:
            cv2.circle(ground, self.players[player_id], 20,  self.team_color, -1)

    def get_player_coords(self, player_obj):
        return self.players[player_obj.id]


class Game:
    def __init__(self):
        self.teamA = Team({'hmin': 0, 'smin': 0, 'vmin': 197,
                          'hmax': 179, 'smax': 255, 'vmax': 255}, (255, 0, 255))
        self.teamB = Team({'hmin': 0, 'smin': 0, 'vmin': 0,
                          'hmax': 179, 'smax': 255, 'vmax': 255}, (0, 255, 0))

        self.trackingIds = {}  # {objid: team}

        self.color_finder = ColorFinder(False)

        # {pos: (x, y), with: ("team": team, "player": player_obj), "kickzone": "color"}
        self.ball_data = {"pos": (720, 720), "with": {
            "team": None, "player": None}, "kickzone": None}

        self.virtual_ground_radius = 360
        self.goal_radius = 50
        self.center = (self.virtual_ground_radius, self.virtual_ground_radius)
        self.red_radius = self.virtual_ground_radius
        self.yellow_radius = self.virtual_ground_radius * 2 // 3
        self.blue_radius = self.virtual_ground_radius // 3

    def generate_football_map(self, image, objects, is_tracking_on):
        # Draws the football ground:
        ground = np.zeros(shape=(self.virtual_ground_radius*2,
                          self.virtual_ground_radius*2, 3), dtype=np.uint8)

        cv2.circle(
            ground, self.center, self.red_radius, (0, 0, 255), -1)
        cv2.circle(ground, self.center, self.yellow_radius *
                   2 // 3, (0, 255, 255), -1)
        cv2.circle(ground, self.center, self.blue_radius // 3, (255, 0, 50), -1)
        cv2.rectangle(ground, (self.center[0]-2, self.center[1]-10),
                      (self.center[0]+2, self.center[1]+10), (255, 255, 255), -1)

        # Draws the players on the ground:
        self.teamA.clear_players()
        self.teamB.clear_players()

        for obj in objects.object_list:
            if (render_object(obj, is_tracking_on)):
                if obj.id not in self.trackingIds:
                    try:
                        if obj.label == sl.OBJECT_CLASS.PERSON:
                            # If object is not being tracked and is a person, figure out their team
                            team = self.determine_team(image, obj)
                            team.add_player(
                                obj, self.virtual_ground_radius*2, self.virtual_ground_radius*2)

                            self.trackingIds[obj.id] = team

                            # Determine if player has ball and update the ball's data
                            self.update_ball_data(obj, team)

                        else:
                            self.ball_data["pos"] = generate_2d_obj_position(
                                obj, self.virtual_ground_radius*2, self.virtual_ground_radius*2)
                            self.trackingIds[obj.id] = None

                    except Exception as e:
                        print(e)

                else:
                    if obj.label == sl.OBJECT_CLASS.PERSON:
                        # If object is being tracked and is a person, add them to their previously determined team
                        team = self.trackingIds[obj.id]
                        team.add_player(
                            obj, self.virtual_ground_radius*2, self.virtual_ground_radius*2)

                        # Determine if player has ball and update the ball's data
                        self.update_ball_data(obj, team)

                    else:
                        self.ball_data["pos"] = generate_2d_obj_position(
                            obj, self.virtual_ground_radius*2, self.virtual_ground_radius*2)

        # Draws each teams players on the ground
        self.teamA.update_football_map(ground)
        self.teamB.update_football_map(ground)

        try:
            # check for goal
            if euclidean_distance(self.ball_data["pos"], self.center) < self.goal_radius:
                if self.ball_data["with"]["team"]:
                    if self.ball_data["kickzone"] == "red":
                        self.ball_data["with"]["team"].add_score(3)
                    elif self.ball_data["kickzone"] == "yellow":
                        self.self.ball_data["with"]["team"].add_score(2)
                    self.ball_data["with"]["team"] = None
                    self.ball_data["with"]["player"] = None
                    
            cv2.circle(ground, self.ball_data["pos"], 10, (255, 255, 255), -1)
            cv2.circle(ground, self.ball_data["with"]["team"].get_player_coords(
                self.ball_data["with"]["player"]), 40, (132, 241, 48), -1)

        except Exception as e:
            pass
            # print(repr(e))
        return ground

    def generate_scoreboard(self):
        window = np.zeros(shape=(self.virtual_ground_radius*2,
                          self.virtual_ground_radius*2, 3), dtype=np.uint8)
        cv2.putText(window, "Scoreboard", (150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(window, "Ball With: Team A" if self.ball_data["with"]["team"] == self.teamA else "Ball With: Team B", (
            10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(window, "Team A: " + str(self.teamA.get_score()),
                    (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(window, "Team B: " + str(self.teamB.get_score()),
                    (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        return window

    def determine_team(self, image, obj):
        top_left_corner = self.cvt(obj.bounding_box_2d[0])
        top_right_corner = self.cvt(obj.bounding_box_2d[1])
        bottom_right_corner = self.cvt(obj.bounding_box_2d[2])
        bottom_left_corner = self.cvt(obj.bounding_box_2d[3])

        roi_height = int(top_right_corner[0] - top_left_corner[0])
        roi_width = int(bottom_left_corner[1] - top_left_corner[1])
        roi = image[int(top_left_corner[1]):int(
            top_left_corner[1] + roi_width), int(top_left_corner[0]):int(top_left_corner[0] + roi_height)]

        _, maskA = self.color_finder.update(roi, self.teamA.get_color())
        _, maskB = self.color_finder.update(roi, self.teamB.get_color())
        # cv2.imshow("maskA", maskA)
        # cv2.imshow("maskB", maskB)

        if (maskA.sum() > maskB.sum()):
            return self.teamA
        else:
            return self.teamB

    def update_ball_data(self, player_obj, team):
        player_2d_coords = team.get_player_coords(player_obj)
        if euclidean_distance(player_2d_coords, self.ball_data["pos"]) < 80:
            self.ball_data["with"]["team"] = team
            self.ball_data["with"]["player"] = player_obj
            center_distance = euclidean_distance(self.center, player_2d_coords)
            if center_distance <= self.red_radius and center_distance > self.yellow_radius:
                self.ball_data["kickzone"] = "red"
            elif center_distance <= self.yellow_radius and center_distance > self.blue_radius:
                self.ball_data["kickzone"] = "yellow"
            elif center_distance <= self.blue_radius:
                self.ball_data["kickzone"] = "blue"

    def cvt(self, pt):
        out = [pt[0], pt[1]]
        return out


def interpolate(x, fromrange, torange):
    frac = (x - fromrange[0]) / (fromrange[1] - fromrange[0])
    return ((torange[1] - torange[0]) * frac) + torange[0]


def generate_2d_obj_position(obj, window_width=720, window_height=720, real_width=8, real_depth=8):
    x_pos = interpolate(
        obj.position[0], (-real_width/2, real_width/2), (0, window_width))
    y_pos = interpolate(abs(obj.position[2]), (
        0, real_depth), (window_height, 0))
    return int(x_pos), int(y_pos)


def render_object(object_data, is_tracking_on):
    if is_tracking_on:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
    else:
        return ((object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF))

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)