"""
Simplified Warehouse GridWorld Simulation
-----------------------------
A simulation of a robot navigating a warehouse to collect and deliver items.

This module implements a grid-based environment where a robot can:
- Navigate through a warehouse with walls and obstacles
- Pick up items from specific locations
- Deliver all items to a single truck
"""

import pygame
import sys
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


class Direction(Enum):
    """Enumeration of possible movement directions."""
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass(frozen=True)
class Position:
    """Represents a 2D position in the grid.
    
    This class is immutable (frozen) to allow it to be hashable
    and used as a dictionary/set key.
    """
    x: int
    y: int

    def __add__(self, other):
        """Add two positions together."""
        return Position(self.x + other.x, self.y + other.y)
    
    def as_tuple(self):
        """Convert position to tuple format."""
        return (self.x, self.y)


class Colors:
    """Standard colors used in the visualization."""
    WHITE = (255, 255, 255)
    LIGHT_GRAY = (240, 240, 240)
    MEDIUM_GRAY = (200, 200, 200)
    DARK_GRAY = (100, 100, 100)
    BLACK = (0, 0, 0)
    ROBOT_BODY = (50, 90, 130)
    ROBOT_ACCENT = (120, 170, 210)
    SUCCESS_COLOR = (100, 200, 100)
    ERROR_COLOR = (255, 80, 80)
    GRID_LINE = (230, 230, 230)
    TEXT_PRIMARY = (60, 60, 70)
    HIGHLIGHT = (255, 240, 200)  # Light yellow highlight
    ITEM_HIGHLIGHT = (220, 240, 255)  # Light blue highlight for items


class NotificationSystem:
    """Manages visual feedback notifications for the simulation."""
    
    def __init__(self):
        self.notifications = {}  # e.g., {"T1": ("success", frame_count)}
        self.permanent_success = set()  # Trucks that have had successful deliveries
        self.frame_count = 0
        self.notification_duration = 30  # frames for temporary notifications
        
    def add_notification(self, entity_id: str, status: str, permanent: bool = False):
        """
        Add a new notification for an entity.
        
        Args:
            entity_id: ID of the entity (e.g., "T1")
            status: Type of notification (e.g., "success", "error")
            permanent: If True, the notification will persist permanently
        """
        self.notifications[entity_id] = (status, self.frame_count)
        if permanent and status == "success":
            self.permanent_success.add(entity_id)
        
    def update(self):
        """Update notification states and remove expired temporary ones."""
        self.frame_count += 1
        expired = []
        
        for entity_id, (status, start_frame) in self.notifications.items():
            # Skip permanent success notifications
            if entity_id in self.permanent_success:
                continue
                
            if self.frame_count - start_frame >= self.notification_duration:
                expired.append(entity_id)
                
        for entity_id in expired:
            del self.notifications[entity_id]
            
    def get_notification(self, entity_id: str) -> Optional[str]:
        """Get active notification status for an entity if it exists."""
        # Permanent success notifications take precedence
        if entity_id in self.permanent_success:
            return "success"
            
        # Otherwise check temporary notifications
        if entity_id in self.notifications:
            status, _ = self.notifications[entity_id]
            return status
        return None


class WarehouseLayout:
    """Manages the physical layout of the warehouse including walls and objects."""
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.walls = self._create_walls()
        self.object_positions = self._place_objects()
        
    def _create_walls(self) -> Set[Tuple[Position, Direction]]:
        """Create walls and barriers in the warehouse."""
        walls = set()
        
        # Horizontal walls
        for x in range(self.cols):
            for y in [0, 3, 6]:
                walls.add((Position(x, y), Direction.DOWN))
                walls.add((Position(x, y + 2), Direction.UP))
                
        # Vertical walls
        for y in range(self.rows):
            for x in [0, 3, 6, 9]:
                walls.add((Position(x, y), Direction.LEFT))
                walls.add((Position(x + 2, y), Direction.RIGHT))

        # Internal shelves/obstacles
        for x in range(2, 10, 2):
            walls.add((Position(x, 4), Direction.DOWN))
            walls.add((Position(x, 3), Direction.UP))

        # Doorways (removing walls to create passages)
        doorways = []
        # Horizontal doorways
        for x in [1, 4, 7, 10]:
            doorways.append((Position(x, 5), Direction.UP))
            doorways.append((Position(x, 6), Direction.DOWN))
        
        for x in [1, 10]:
            doorways.append((Position(x, 2), Direction.UP))
            doorways.append((Position(x, 3), Direction.DOWN))
        
        # Vertical doorways
        for y in [1, 7]:
            for x in [2, 5, 8]:
                doorways.append((Position(x, y), Direction.RIGHT))
                doorways.append((Position(x + 1, y), Direction.LEFT))
        
        for doorway in doorways:
            if doorway in walls:
                walls.remove(doorway)
                
        return walls
    
    def _place_objects(self) -> Dict[Tuple[int, int], str]:
        """Define positions of all objects in the warehouse."""
        return {
            (1, 1): '🍎',  # Apple
            (2, 6): '🥦',  # Lettuce
            (4, 1): '🥩',  # Steak
            (5, 7): '🍞',  # Bread
            (3, 3): '🥛',  # Milk
            (7, 2): '🥣',  # Cereal
            (6, 5): '🍌',  # Banana
            (11, 4): 'T1',  # Single Truck (centered)
        }
        
    def is_valid_move(self, position: Position, direction: Direction) -> bool:
        """Check if a move from the given position in the given direction is valid."""
        # Check if we'd hit a wall
        wall_check = (position, direction)
        if wall_check in self.walls:
            return False
            
        # Calculate new position after move
        dx, dy = {
            Direction.UP: (0, 1),
            Direction.DOWN: (0, -1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0)
        }[direction]
        
        new_x, new_y = position.x + dx, position.y + dy
        
        # Check if within grid bounds
        if not (0 <= new_x < self.cols and 0 <= new_y < self.rows):
            return False
            
        return True
    
    def get_object_at(self, position: Position) -> Optional[str]:
        """Get the object at the given position, if any."""
        return self.object_positions.get((position.x, position.y))


class InventorySystem:
    """Manages the robot's inventory and delivery tracking."""
    
    def __init__(self):
        # Item -> whether it's in inventory
        self.inventory = {
            "apple": False,
            "lettuce": False, 
            "steak": False, 
            "bread": False, 
            "milk": False, 
            "cereal": False, 
            "banana": False
        }
        
        # Items successfully delivered
        self.delivered_items = set()
        
        # Only one truck that accepts all items
        self.truck_requirements = {
            "T1": ["apple", "lettuce", "steak", "bread", "milk", "cereal", "banana"]
        }
        
        # For convenience, create a reverse mapping (item -> truck)
        self.delivery_goals = {}
        for truck, items in self.truck_requirements.items():
            for item in items:
                self.delivery_goals[item] = truck
        
        # Emoji -> item name mapping
        self.emoji_to_item = {
            '🍎': 'apple', 
            '🥦': 'lettuce', 
            '🥩': 'steak',
            '🍞': 'bread', 
            '🥛': 'milk', 
            '🥣': 'cereal', 
            '🍌': 'banana'
        }
    
    def pick_up_item(self, emoji: str) -> Optional[str]:
        """Attempt to pick up an item represented by emoji. Returns item name if successful."""
        if emoji in self.emoji_to_item:
            item = self.emoji_to_item[emoji]
            self.inventory[item] = True
            return item
        return None
    
    def get_carried_items(self) -> List[str]:
        """Get a list of all items currently being carried."""
        return [item for item, carrying in self.inventory.items() if carrying]
        
    def evaluate_delivery(self, truck_id: str) -> Tuple[bool, List[str]]:
        """
        Evaluate a delivery attempt for the given truck.
        
        Returns:
            Tuple of (success, delivered_items)
        """
        # Check which items we're carrying
        carried_items = self.get_carried_items()
        
        # Deliver any carried items (simplified - no exact match required)
        if carried_items:
            delivered = []
            for item in carried_items:
                self.inventory[item] = False
                self.delivered_items.add(item)
                delivered.append(item)
            return True, delivered
        
        return False, []
    
    def get_delivery_status(self, item: str) -> str:
        """Get the current status of an item."""
        if item in self.delivered_items:
            return "✓"  # Delivered
        elif self.inventory[item]:
            return "◉"  # In inventory
        else:
            return "◯"  # Not collected
            

class GridWorldRenderer:
    """Handles all rendering for the GridWorld simulation."""
    
    def __init__(self, width: int, height: int, layout: WarehouseLayout):
        self.width = width
        self.height = height
        self.layout = layout
        self.cell_size = width // layout.cols
        
        # Add additional vertical space for the title bar
        title_bar_height = 40
        total_height = height + title_bar_height
        
        # Initialize Pygame components
        pygame.init()
        
        # Set window title with a more academic title
        self.screen = pygame.display.set_mode((width, total_height))
        pygame.display.set_caption("Simplified Warehouse Logistics Simulation")
        
        # Use higher quality fonts
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14)
        self.title_font = pygame.font.SysFont('Arial', 18, bold=True)
        
        # For emojis, we need to make sure we use a font that supports them
        try:
            self.emoji_font = pygame.font.SysFont('Segoe UI Emoji', 24)
        except:
            # Fallback to other emoji-compatible fonts
            try:
                self.emoji_font = pygame.font.SysFont('Apple Color Emoji', 24)
            except:
                try:
                    self.emoji_font = pygame.font.SysFont('Noto Color Emoji', 24)
                except:
                    # Last resort fallback
                    self.emoji_font = pygame.font.SysFont('Arial Unicode MS', 24)
        
        # Load robot components once
        self.create_robot_parts()
        
        # Vertical offset for rendering the grid below the title bar
        self.grid_y_offset = title_bar_height
        
    def create_robot_parts(self):
        """Pre-create robot visual components."""
        size = self.cell_size
        half = size // 2
        quarter = size // 4
        
        # Create a surface for the robot
        self.robot_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Main body (rounded rectangle)
        body_rect = pygame.Rect(quarter, quarter, half, half)
        pygame.draw.rect(self.robot_surface, Colors.ROBOT_BODY, body_rect, border_radius=5)
        
        # Robot tracks/wheels
        wheel_height = quarter // 2
        left_wheel = pygame.Rect(quarter - 2, half - wheel_height // 2, quarter // 2, wheel_height)
        right_wheel = pygame.Rect(size - quarter - quarter // 2 + 2, half - wheel_height // 2, quarter // 2, wheel_height)
        pygame.draw.rect(self.robot_surface, Colors.DARK_GRAY, left_wheel, border_radius=2)
        pygame.draw.rect(self.robot_surface, Colors.DARK_GRAY, right_wheel, border_radius=2)
        
        # Robot "eye"/sensor
        eye_size = quarter // 2
        eye_pos = (half, half - quarter // 2)
        pygame.draw.circle(self.robot_surface, Colors.ROBOT_ACCENT, eye_pos, eye_size)
        pygame.draw.circle(self.robot_surface, Colors.WHITE, eye_pos, eye_size // 2)
        
        # Arm
        arm_width = 3
        pygame.draw.line(self.robot_surface, Colors.ROBOT_ACCENT, 
                         (half, half), (half + quarter, half + quarter), arm_width)
        pygame.draw.circle(self.robot_surface, Colors.ROBOT_ACCENT, 
                           (half + quarter, half + quarter), 4)
        
    def draw_grid(self):
        """Draw the base grid lines."""
        # Fill background with light gray
        self.screen.fill(Colors.LIGHT_GRAY)
        
        # Draw title bar
        title_bar = pygame.Rect(0, 0, self.width, self.grid_y_offset)
        pygame.draw.rect(self.screen, Colors.ROBOT_BODY, title_bar)
        
        # Draw simulation title
        title_text = self.title_font.render("Simplified Warehouse Logistics Simulation", True, Colors.WHITE)
        self.screen.blit(title_text, (10, 10))
        
        # Draw grid lines (shifted down by title bar height)
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, Colors.GRID_LINE, 
                            (x, self.grid_y_offset), 
                            (x, self.height + self.grid_y_offset))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, Colors.GRID_LINE, 
                            (0, y + self.grid_y_offset), 
                            (self.width, y + self.grid_y_offset))
            
    def draw_walls(self):
        """Draw all walls in the warehouse."""
        for (pos, direction) in self.layout.walls:
            sx = pos.x * self.cell_size
            # Convert y-coordinate to screen space (inverted y-axis)
            sy = (self.layout.rows - 1 - pos.y) * self.cell_size + self.grid_y_offset
            
            # Draw wall segment based on direction
            if direction == Direction.UP:
                pygame.draw.line(self.screen, Colors.BLACK, (sx, sy), 
                                 (sx + self.cell_size, sy), 3)
            elif direction == Direction.DOWN:
                pygame.draw.line(self.screen, Colors.BLACK, (sx, sy + self.cell_size), 
                                 (sx + self.cell_size, sy + self.cell_size), 3)
            elif direction == Direction.LEFT:
                pygame.draw.line(self.screen, Colors.BLACK, (sx, sy), 
                                 (sx, sy + self.cell_size), 3)
            elif direction == Direction.RIGHT:
                pygame.draw.line(self.screen, Colors.BLACK, (sx + self.cell_size, sy), 
                                 (sx + self.cell_size, sy + self.cell_size), 3)
    
    def draw_objects(self, notification_system: NotificationSystem):
        """Draw all objects in the warehouse."""
        for (x, y), label in self.layout.object_positions.items():
            # Convert y-coordinate to screen space
            pos_y = self.layout.rows - 1 - y
            rect = pygame.Rect(x * self.cell_size, 
                               pos_y * self.cell_size + self.grid_y_offset, 
                               self.cell_size, self.cell_size)
            
            # Draw item cell background highlighting
            if not label.startswith("T"):
                # Items have a light blue background
                pygame.draw.rect(self.screen, Colors.ITEM_HIGHLIGHT, rect)
            
            # Apply notification highlights for trucks
            if label.startswith("T"):
                # Truck cell has slightly different background
                pygame.draw.rect(self.screen, Colors.MEDIUM_GRAY, rect)
                
                # Check notification status
                status = notification_system.get_notification(label)
                if status == "success":
                    # Draw a more subtle green background for successful delivery
                    pygame.draw.rect(self.screen, Colors.SUCCESS_COLOR, rect)
                elif status == "error":
                    # Draw red background for error
                    pygame.draw.rect(self.screen, Colors.ERROR_COLOR, rect)
            
            # Draw object label with a slight drop shadow for better visibility
            if label.startswith("T"):
                # Trucks get a more professional look
                pygame.draw.rect(self.screen, Colors.DARK_GRAY, 
                                rect.inflate(-self.cell_size//3, -self.cell_size//3),
                                border_radius=5)
                text = self.title_font.render(label, True, Colors.WHITE)
            else:
                # Use emoji font for item emojis
                text = self.emoji_font.render(label, True, Colors.BLACK)
            
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_robot(self, position: Position):
        """Draw the robot at the specified position."""
        # Convert y-coordinate to screen space
        pos_y = self.layout.rows - 1 - position.y
        rect = pygame.Rect(position.x * self.cell_size, 
                          pos_y * self.cell_size + self.grid_y_offset, 
                          self.cell_size, self.cell_size)
        
        # Blit the pre-created robot surface
        self.screen.blit(self.robot_surface, rect)
    
    def draw_status(self, inventory: InventorySystem):
        """Draw information about progress and simulation title."""
        # Draw progress bar at bottom
        items_delivered = len(inventory.delivered_items)
        total_items = len(inventory.inventory)
        
        # Progress bar background
        bar_height = 24
        bar_y = self.height + self.grid_y_offset - bar_height - 10
        bar_width = self.width - 40
        pygame.draw.rect(self.screen, Colors.MEDIUM_GRAY, pygame.Rect(20, bar_y, bar_width, bar_height), border_radius=5)
        
        # Progress bar fill
        if total_items > 0:
            fill_width = int((items_delivered / total_items) * bar_width)
            pygame.draw.rect(self.screen, Colors.SUCCESS_COLOR, 
                            pygame.Rect(20, bar_y, fill_width, bar_height), border_radius=5)
        
        # Progress text
        progress_text = self.font.render(f"Progress: {items_delivered}/{total_items} items delivered", 
                                    True, Colors.TEXT_PRIMARY)
        self.screen.blit(progress_text, (30, bar_y + 4))
            
    def render_frame(self, robot_pos: Position, inventory: InventorySystem, 
                     notification_system: NotificationSystem):
        """Render a complete frame of the simulation."""
        self.draw_grid()
        self.draw_walls()
        self.draw_objects(notification_system)
        self.draw_robot(robot_pos)
        self.draw_status(inventory)
        pygame.display.flip()


class GridWorldSimulation:
    """Main class that manages the warehouse simulation."""
    
    def __init__(self, width=960, height=720, rows=9, cols=12):
        self.layout = WarehouseLayout(rows, cols)
        self.robot_pos = Position(1, 1)
        self.inventory = InventorySystem()
        self.notification_system = NotificationSystem()
        self.renderer = GridWorldRenderer(width, height, self.layout)
        self.running = True
        
    def handle_input(self):
        """Process user input from keyboard."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            if event.type == pygame.KEYDOWN:
                direction_map = {
                    pygame.K_UP: Direction.UP,
                    pygame.K_DOWN: Direction.DOWN,
                    pygame.K_LEFT: Direction.LEFT,
                    pygame.K_RIGHT: Direction.RIGHT,
                    pygame.K_ESCAPE: None  # Special case for quitting
                }
                
                if event.key in direction_map:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    else:
                        self.move_robot(direction_map[event.key])
    
    def move_robot(self, direction: Direction):
        """Attempt to move the robot in the specified direction."""
        if self.layout.is_valid_move(self.robot_pos, direction):
            # Calculate new position
            dx, dy = {
                Direction.UP: (0, 1),
                Direction.DOWN: (0, -1),
                Direction.LEFT: (-1, 0),
                Direction.RIGHT: (1, 0)
            }[direction]
            
            self.robot_pos = Position(self.robot_pos.x + dx, self.robot_pos.y + dy)
            self._check_interaction()
    
    def _check_interaction(self):
        """Check for interactions at the robot's current position."""
        object_at_pos = self.layout.get_object_at(self.robot_pos)
        
        if object_at_pos in self.inventory.emoji_to_item:
            # Pick up item
            self.inventory.pick_up_item(object_at_pos)
            
        elif object_at_pos and object_at_pos.startswith("T"):
            truck_id = object_at_pos
            
            # Attempt delivery
            success, delivered_items = self.inventory.evaluate_delivery(truck_id)
            
            if success and delivered_items:
                # Show successful delivery notification
                self.notification_system.add_notification(truck_id, "success", permanent=True)
            else:
                # Error - nothing to deliver
                self.notification_system.add_notification(truck_id, "error")
    
    def run(self):
        """Run the main simulation loop."""
        fps = 10  # Frames per second
        
        while self.running:
            self.handle_input()
            self.notification_system.update()
            self.renderer.render_frame(
                self.robot_pos, 
                self.inventory,
                self.notification_system
            )
            self.renderer.clock.tick(fps)
            
        # Clean up
        pygame.quit()
        sys.exit()


def main():
    """Entry point for the simulation."""
    simulation = GridWorldSimulation()
    simulation.run()


if __name__ == "__main__":
    main()