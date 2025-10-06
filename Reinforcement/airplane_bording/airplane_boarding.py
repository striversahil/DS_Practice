import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.envs.registration import register
import numpy as np

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
        
        if passenger.is_carrying_baggage:
            passenger.status = PassengerStatus.STANDING
            passenger.is_carrying_baggage = False  # Baggage is stowed
            return False  # Passenger is standing to stow baggage
        
        else:
            self.passenger = passenger
            self.passenger.status = PassengerStatus.SEATED
            return True  # Passenger is seated
        
    def __str__(self):
        return f"{self.seat_no}"

class AirplaneRow:
    def __init__(self, row_num, seats_per_row):
        self.row_num = row_num
        self.seats = [Seat(row_num * seats_per_row + i, row_num) for i in range(seats_per_row)]

    def try_sit_passenger(self, passenger: Passenger):
        # Check if passenger's seat is in this row
        found_seats = list(filter(lambda seats: seats.seat_num == passenger.seat_num, self.seats))

        if found_seats:
            found_seat: Seat = found_seats[0]
            return found_seat.seat_passenger(passenger)

        return False
    
        
class AirplaneBoardingEnv(gym.Env):
    """Custom Environment that follows gym interface. This is a simple example of an airplane boarding simulation."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self , render_mode=None , seats_per_row=6 , num_rows=30):

        self.seats_per_row = seats_per_row
        self.num_rows = num_rows
        self.no_of_seats = self.seats_per_row * self.num_rows
        self.render_mode = render_mode
        
        # Resets the environment to an initial state
        self.reset()

        self.action_space = spaces.Discrete(self.num_rows)  # Action is to select a row to board from

        self.observation_space = spaces.Box(
            low=-1,
            high=self.no_of_seats ,
            shape=(self.no_of_seats * 2 ,),
            dtype=np.int32,
        )
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the environment state
        self.lobby = Lobby(self.num_rows , self.seats_per_row)
        self.airplane = [AirplaneRow(i , self.seats_per_row) for i in range(1 , self.num_rows + 1)]
        self.boarding_area = BoardingArea(self.num_rows)

        self.render()

        return self._getobservation() , {}
    
    def _getobservation(self):
        """Get the Observation of the current state."""
        observation = []

        for passenger in self.boarding_area.line:
            if passenger is None:
                observation.append(-1)
                observation.append(-1)
            else:
                observation.append(passenger.seat_no)
                observation.append(passenger.status.value)

        return np.array(observation , dtype=np.int32)

    # Takes an action and returns the next state, reward, observation, and info

    def step(self, row_num):
        assert row_num>=0 and row_num<self.num_of_rows, f"Invalid row number {row_num}"

        reward = 0

        passenger = self.lobby.remove_passenger(row_num)
        self.boarding_line.add_passenger(passenger)

        # If there are passengers in the lobby, move the line once
        if self.lobby.count_passengers()>0:
            self._move()
            reward = self._calculate_reward()
        else:
            # No more passengers in the lobby, so no more actions to choose from, move the line until all passengers are seated
            while self.is_onboarding():
                self._move()
                reward += self._calculate_reward()

        if self.is_onboarding():
            terminated = False
        else:
            terminated = True

        # Gym requires returning the observation, reward, terminated, truncated, and info dictionary.
        return self._get_observation(), reward, terminated, False, {}
    
    def _move(self):

        for row_num, passenger in enumerate(self.boarding_line.line):
            if passenger is None:
                continue

            # If outside of airplane's aisle
            if row_num >= len(self.airplane_rows):
                break

            # Try to sit passenger, if successful, remove from line
            if self.airplane_rows[row_num].try_sit_passenger(passenger):
                self.boarding_line.line[row_num] = None

        # Move line forward
        self.boarding_line.move_forward()

        self.render()


    def _reward(self):
        """Calculate the reward for the current state."""
        reward = 0

        # Reward for each seated passenger
        for airplane_row in self.airplane:
            for seat in airplane_row.seats:
                if seat.passenger is not None and seat.passenger.status == PassengerStatus.SEATED:
                    reward += 1

        # Penalty for each waiting passenger in the boarding area
        reward -= self.boarding_area.no_of_waiting_passengers() * 0.5

        # Penalty for each moving passenger in the boarding area
        reward -= self.boarding_area.no_of_moving_passengers() * 0.2

        return reward

    def is_boarding_complete(self):
        """Check if boarding is complete."""
        if self.lobby.count_passengers() == 0 and not self.boarding_area.is_onboarding():
            return True
        return False

    # Renders the environment to the screen or other modes
    def render(self):
        if self.render_mode is None:
            return
        
        if self.render_mode == "terminal":
            print("\n" + "="*50)
            print("Lobby:")
            for row in self.lobby.lobby_rows:
                print(f"Row {row.row_no:02d}: " + " ".join(str(passenger) for passenger in row.passengers))
            
            print("\nBoarding Area:")
            boarding_area_str = []
            for passenger in self.boarding_area.line:
                if passenger is None:
                    boarding_area_str.append(" . ")
                else:
                    boarding_area_str.append(f"{str(passenger):>2} ")
            print("".join(boarding_area_str))
            
            print("\nAirplane Seating:")
            for airplane_row in self.airplane:
                seats_str = []
                for seat in airplane_row.seats:
                    if seat.passenger is None:
                        seats_str.append(" . ")
                    else:
                        seats_str.append(f"{str(seat.passenger):>2} ")
                print(f"Row {airplane_row.row_num:02d}: " + "".join(seats_str))
            print("="*50 + "\n")

    # This method is used to mask the actions that are allowed
    # This will return True for allowed actions and False for disallowed actions
    def action_masks(self) -> list[bool]:
        mask = []

        for row in self.lobby.lobby_rows:
            if len(row.passengers) == 0:
                mask.append(False)
            else:
                mask.append(True)

        return mask


# Check the validity of the custom environment
def custom_check_env():
    """Check the custom environment using Gym's built-in utility."""

    from gymnasium.utils.env_checker import check_env

    env = gym.make("AirplaneBoarding-v0")
    check_env(env.unwrapped)

    



if __name__ == "__main__":
    # my_check_env()

    env = gym.make('airplane-boarding-v0', num_of_rows=10, seats_per_row=5, render_mode='terminal')

    observation, _ = env.reset()
    terminated = False
    total_reward = 0
    step_count = 0

    while not terminated:
        # Choose random action
        action = env.action_space.sample()

        # Skip action if invalid
        masks = env.unwrapped.action_masks()
        if(masks[action]==False):
            continue

        # Perform action
        observation, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        step_count+=1

        print(f"Step {step_count} Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}\n")

    print(f"Total Reward: {total_reward}")