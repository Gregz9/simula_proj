import sys
import termios
import tty


def read_multiple_characters():
    # Save the current terminal settings
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        # Set the terminal to raw mode
        tty.setraw(sys.stdin)

        # Read multiple characters
        characters = []
        while True:
            char = sys.stdin.read(1)
            characters.append(char)

            # Break the loop if the user presses Enter
            if char == "\n":
                break

    finally:
        # Restore the original terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return "".join(characters)


# Usage
input_string = read_multiple_characters()
print("Input:", input_string)
