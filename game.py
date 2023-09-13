from pygame import mixer, time, event
import time

class Game:
    def __init__(self):
        mixer.init()
        mixer.music.set_volume(1)
        self.load("song.mp3")
        
        self.length = 0

    def play_sound(self):
        mixer.music.play()
        time.sleep(self.length)

    def load(self, song):
        self.length = mixer.Sound(song).get_length()
        mixer.music.load(song)

if __name__ == "__main__":
    game = Game()
    game.load("song.mp3")
    game.play_sound()