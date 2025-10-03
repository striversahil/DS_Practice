import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.envs.registration import register

# Once registered, you can create the environment using: gym.make("AirplaneBoarding-v0")
register(
    id="AirplaneBoarding-v0",
    entry_point="airplane_boarding:AirplaneBoardingEnv", # Module and class name
    # max_episode_steps=200,
)

class PassengerStatus:
    """Class to represent the status of a passenger in the airplane boarding simulation."""
    MOVING = 0
    WAITING = 1
    SEATED = 2
    STANDING = 3


    def __str__(self):
        match (self):
            case PassengerStatus.MOVING:
                return "MOVING"
            case PassengerStatus.WAITING:
                return "WAITING"
            case PassengerStatus.SEATED:
                return "SEATED"
            case PassengerStatus.STANDING:
                return "STANDING"
            
class Passenger:
    """Class to represent a passenger in the airplane boarding simulation."""
    def __init__(self , row_no : int , seat_no : str , is_carrying_baggage : bool = True):
        self.row_no = row_no
        self.seat_no = seat_no
        self.is_carrying_baggage = is_carrying_baggage
        self.status = PassengerStatus.MOVING
    
    def __str__(self):
        return f"{self.seat_no:02d}"
    

class LobbyRow:
    """Class to represent a row in the airplane lobby."""
    def __init__(self , row_no : int , seats_per_row : int):
        self.row_no = row_no
        self.passengers = [Passenger(row_no , row_no * seats_per_row + i) for i in range(1 , seats_per_row + 1)]   # Assuming seat numbers start from 1


class Lobby:
    """Class to represent the airplane lobby."""
    def __init__(self , num_rows : int , seats_per_row : int):
        self.num_rows = num_rows
        self.seats_per_row = seats_per_row
        self.lobby_rows = [LobbyRow(i , self.seats_per_row) for i in range(1 , self.num_rows + 1)]
        
    def remove_passenger(self , row_num: int):
        """Remove a passenger from the lobby row."""
        if self.lobby_rows[row_num - 1].passengers:
            return self.lobby_rows[row_num - 1].passengers.pop(0)
        return None
    
    def count_passengers(self):
        """Count the total number of passengers in the lobby."""
        return sum(len(row.passengers) for row in self.lobby_rows)
    

class BoardingArea:
    """Class to represent the boarding area of the airplane."""
    def __init__(self , num_rows : int):
        self.num_rows = num_rows   # Number of rows in the boarding area
        self.line = [None for _ in range(num_rows)]  # None indicates empty space
        
    def add_passenger(self , passenger : Passenger):
        """Add a passenger to the boarding area."""
        self.line.append(passenger)

    def is_onboarding(self):
        """Check if Passengers are still onboarding."""
        if len(self.line) > 0 and all(passenger is None for passenger in self.line):
            return False
        return True
    
    def no_of_waiting_passengers(self):
        """Count the number of waiting passengers in the boarding area."""
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.WAITING:
                count += 1
        return count
    
    def no_of_moving_passengers(self):
        """Count the number of moving passengers in the boarding area."""
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.MOVING:
                count += 1
        return count
    
    def move_forward(self):
        """Move passengers forward in the boarding area."""
        for i, passenger in enumerate(self.line):
            # Skip, if no passenger in that spot or
            #   passenger is at the front of the line or
            #   passenger is stowing luggage
            if passenger is None or i==0 or passenger.status == PassengerStatus.STANDING:
                # continue to next passenger
                continue

            # Move passenger forward, if no one is blocking the way 
            if (passenger.status == PassengerStatus.WAITING or passenger.status == PassengerStatus.MOVING) and self.line[i-1] is None:
                passenger.status = PassengerStatus.MOVING
                self.line[i-1] = passenger
                self.line[i] = None
            else:
                passenger.status = PassengerStatus.WAITING

        # Truncate the empty spots at the end of the line
        for i in range(len(self.line)-1, self.num_rows-1, -1):
            if self.line[i] is None:
                self.line.pop(i)

        
class Seat:
    """Class to represent a seat in the airplane."""
    def __init__(self , row_no : int , seat_no : str):
        self.row_no = row_no
        self.seat_no = seat_no
        self.passenger = None  # No passenger initially

    def seat_passenger(self , passenger : Passenger):
        """Seat a passenger in the seat."""
        
        assert passenger.row_no == self.row_no , "Passenger row number does not match seat row number"
        assert passenger.seat_no == self.seat_no , "Passenger seat number does not match seat number"
        assert self.passenger is None , "Seat is already occupied"
    
        
class AirplaneBoardingEnv(gym.Env):
    """Custom Environment that follows gym interface. This is a simple example of an airplane boarding simulation."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self , render_mode=None):
        
        # Resets the environment to an initial state
        self.reset()

    def reset(self, seed=None, options=None):

        super().reset()

    # Takes an action and returns the next state, reward, observation, and info
    def step(self, action):
        return super().step(action)

    # Renders the environment to the screen or other modes
    def render(self):
        pass



# Check the validity of the custom environment
def custom_check_env():
    """Check the custom environment using Gym's built-in utility."""

    from gymnasium.utils.env_checker import check_env

    env = gym.make("AirplaneBoarding-v0")
    check_env(env.unwrapped)

    



if __name__ == "__main__":
    custom_check_env()