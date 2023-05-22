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
        self.has_ball = False
        self.players = {} # {playerobj: 2d coordinates}

    def get_has_ball(self):
        return self.has_ball
    
    def set_has_ball(self, has_ball):
        self.has_ball = has_ball

    def get_color(self):
        return self.color

    def get_score(self):
        return self.score

    def goal(self):
        self.score += 1

    def get_players(self):
        return self.player

    def add_player(self, player):
        self.players[player] = generate_2d_obj_position(player)

    def remove_player(self, player):
        self.players.pop(player)

    def clear_players(self):
        self.players.clear()

    def update_football_map(self, ground):
        for player in self.players:
            cv2.circle(ground, self.players[player], 20,  self.team_color, -1)
            
    def check_ball(self, ball_pos):
        for player in self.players:
            if math.sqrt((self.players[player][0] - ball_pos[0])**2 + (self.players[player][1] - ball_pos[1])**2) < 80:
                return True
        return False

class Game:
    def __init__(self):
        self.teamA = Team({'hmin': 0, 'smin': 0, 'vmin': 197,
                          'hmax': 179, 'smax': 255, 'vmax': 255}, (255, 0, 255))
        self.teamB = Team({'hmin': 0, 'smin': 0, 'vmin': 0,
                          'hmax': 179, 'smax': 255, 'vmax': 255}, (0, 255, 0))
        
        self.trackingIds = {}

        self.color_finder = ColorFinder(False)

    def generate_football_map(self, image, objects, is_tracking_on, width=720, height=720, radius=360):
        center = (width//2, height//2)

        ground = np.zeros(shape=(height, width, 3), dtype=np.uint8)

        cv2.circle(
            ground, (int(ground.shape[1]/2), int(ground.shape[0]/2)), radius, (0, 0, 255), -1)
        cv2.circle(ground, (int(
            ground.shape[1]/2), int(ground.shape[0]/2)), radius * 2 // 3, (0, 255, 255), -1)
        cv2.circle(ground, (int(
            ground.shape[1]/2), int(ground.shape[0]/2)), radius // 3, (255, 0, 50), -1)
        cv2.rectangle(ground, (center[0]-2, center[1]-10),
                      (center[0]+2, center[1]+10), (255, 255, 255), -1)

        self.teamA.clear_players()
        self.teamB.clear_players()
        for obj in objects.object_list:
            if (render_object(obj, is_tracking_on)):
                if obj.id not in self.trackingIds:
                    try:
                        if obj.label == sl.OBJECT_CLASS.PERSON:
                            team = self.determine_team(image, obj)
                            team.add_player(obj)
                            self.trackingIds[obj.id] = team

                        else:
                            ball_pos = generate_2d_obj_position(obj)
                            self.trackingIds[obj.id] = None

                    except Exception as e:
                        print(e)

                else:
                    if obj.label == sl.OBJECT_CLASS.PERSON:
                        self.trackingIds[obj.id].add_player(obj)
                    else:
                        ball_pos = generate_2d_obj_position(obj)

        self.teamA.update_football_map(ground)
        self.teamB.update_football_map(ground)
        try:
            cv2.circle(ground, ball_pos, 10, (255, 255, 255), -1)
            if self.teamA.check_ball(ball_pos):
                self.teamA.set_has_ball(True)
                self.teamB.set_has_ball(False)
            elif self.teamB.check_ball(ball_pos):
                self.teamA.set_has_ball(False)
                self.teamB.set_has_ball(True)
            
            # check for goal
            goal_radius = 50
            if math.sqrt((ball_pos[0] - center[0])**2 + (ball_pos[1] - center[1])**2) < goal_radius:
                if self.teamA.get_has_ball():
                    self.teamA.goal()
                    self.teamA.set_has_ball(False)
                elif self.teamB.get_has_ball():
                    self.teamB.goal()
                    self.teamB.set_has_ball(False)
                    
                
                
        except Exception as e:
            pass
        return ground

    def generate_scoreboard(self, width=540, height=720):
        window = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        cv2.putText(window, "Scoreboard", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(window, "Ball With: Team A" if self.teamA.get_has_ball(
        ) else "Ball With: Team B", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(window, "Team A: " + str(self.teamA.get_score()),
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(window, "Team B: " + str(self.teamB.get_score()),
                    (270, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
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

    def cvt(self, pt):
        out = [pt[0], pt[1]]
        return out


def interpolate(x, fromrange, torange):
    frac = (x - fromrange[0]) / (fromrange[1] - fromrange[0])
    return ((torange[1] - torange[0]) * frac) + torange[0]
    # return (x - fromrange[0]) * (torange[1] - torange[0]) / (fromrange[1] - fromrange[0]) + torange[0]


def generate_2d_obj_position(obj, window_width=720, window_height=720, real_width=8, real_depth=8):
    x_pos = interpolate(
        obj.position[0], [-real_width/2, real_width/2], [0, window_width])
    y_pos = interpolate(abs(obj.position[2]), [
        0, real_depth], [window_height, 0])
    return int(x_pos), int(y_pos)


def render_object(object_data, is_tracking_on):
    if is_tracking_on:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
    else:
        return ((object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF))
