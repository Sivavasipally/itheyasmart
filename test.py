import subprocess

def step1_open_keynote():
    print("Opening keynote")
    apple_script = '''
    tell application "Keynote"
        activate
        set thisDocument to make new document with properties {document theme:theme "White"}
        tell thisDocument
            set base slide of the first slide to master slide "Blank"
        end tell
    end tell
    '''
    result = subprocess.run(["osascript", "-e", apple_script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("Step 1 Error:", result.stderr)
        return False
    print("Step 1 completed: Keynote opened and new document created.")
    return True

def step2_draw_rectangle(size=(300, 200)):
    shapeWidth, shapeHeight = size
    apple_script = f'''
    tell application "Keynote"
        tell document 1
            set docWidth to its width
            set docHeight to its height
            set x to (docWidth - {{{shapeWidth}}}) div 2
            set y to (docHeight - {{{shapeHeight}}}) div 2
            tell slide 1
                set newRectangle to make new shape with properties {{position:{{x, y}}, width:{shapeWidth}, height:{shapeHeight}}}
            end tell
        end tell
    end tell
    '''
    result = subprocess.run(["osascript", "-e", apple_script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("Step 2 Error:", result.stderr)
        return False
    print("Step 2 completed: Rectangle drawn on the slide.")
    return True

def step3_add_text(text="Hello, MCP!"):
    apple_script = f'''
    tell application "Keynote"
        tell document 1
            tell slide 1
                set the object text of the shape 1 to "{text}"
            end tell
        end tell
    end tell
    '''
    result = subprocess.run(["osascript", "-e", apple_script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("Step 3 Error:", result.stderr)
        return False
    print("Step 3 completed: Text added to the rectangle.")
    return True


# Example call
def open_keynote():
    if step1_open_keynote():
        print("Proceed to Step 2")
        if step2_draw_rectangle():
            print("Proceed to Step 3")
            if step3_add_text("Heyy! it works!"):
                print("All steps completed successfully!")
            else:
                print("Error in Step 3")
        else :
            print("Error in Step 2")
    else:
        print("Error in Step 1")


if __name__ == "__main__":
    open_keynote()