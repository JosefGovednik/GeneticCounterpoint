from music21 import *
import random
import numpy as np

# ***************************************************************
# ----------------------- GLOBAL VARS ------------------------------
# ***************************************************************
# new global config variables so you can change the key, scale,
# tempo, time sig, etc.

# num of generations and number of notes to be mutated every evolution step
GENERATIONS = 1000
MUT_AMOUNT = (1, 3)   # low,high of possible notes to be mutated every iteration

# key/scale types
KEY = 'C'
SCALE_TYPE = 'major'

# scale type vals ---
# Scale may look a little weird due to the music21 notation, but think of it like this:
# C   C#  D   D#  E   F   F#  G   G#  A   A#  B
# 0   1   2   3   4   5   6   7   8   9   10  11
# minor is: [0,2,3,5,7,8,10]
# etc.
SCALE_PATRNS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
    'blues': [0, 3, 5, 6, 7, 10],
    'whole_tone': [0, 2, 4, 6, 8, 10],
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}
SCALE_VALS = SCALE_PATRNS[SCALE_TYPE] 

# compose vals ---
TIME_SIG = '4/4'
TEMPO = 120

# measure vals ---
MEASURES = 4
BEATS_PER_MEASURE = 4
TOTAL_NOTES = MEASURES * BEATS_PER_MEASURE

# DEFINE THE VOICE CENTER!!!!!
# this will basically decide the "center" point for all of the voice lines.
# Eg. C4 = 60, D4 = 62, E4 = 64, F4 = 65, etc.. If you know MIDI keyboard vals this'll make sense.
VOICE_CENTER = 66    # F#4 is the default center point for the C scale

# this is the total span that all voices can travel combined
# I REALLY WOULDNT TOUCH THIS VARIABLE****************
# this can get pretty volatile and weird and impossible to play chords so be careful
ALL_SPAN = 12       # 12 == 1 octave

# range for each voice line itself. this isnt as volatile as the one above,
# but can still lead to unexpected behavior I haven't accounted for.
INDV_SPAN = 5

# this will then auto calculate the range of each voice based on the center and indv spans.
VOICES_RANGE = {
    'melody': (VOICE_CENTER+1, VOICE_CENTER+1 + INDV_SPAN),
    'upper': (VOICE_CENTER-2, VOICE_CENTER-2 + INDV_SPAN),
    'middle': (VOICE_CENTER-4, VOICE_CENTER-4 + INDV_SPAN),
    'bass': (VOICE_CENTER-6, VOICE_CENTER-6 + INDV_SPAN),
}

# this auto forms a major triad to resolve on based on the bass bottom bass note.
# so it works upwards to form a major triad basically.
END_NOTES = {
    'melody': VOICES_RANGE['bass'][0]+ 12,    # mirror the root but one octave raised
    'upper': VOICES_RANGE['bass'][0] + 7,      # fifth to bass
    'middle': VOICES_RANGE['bass'][0] + 4,     # third to bass
    'bass': VOICES_RANGE['bass'][0],           # bass root note
}

# max amount of length that the alt pattern checker will check.
# 2 == ABAB max
# 3 == ABAB and ABCABC and so on...
ALT_MAX_CHECK = 4

# --- HORIZONTAL FITNESS WEIGHTS -----------------
SAME_NOTE_WGT = 1
UNIQUE_WGT = 4
STEPWISE_WGT = 4
SM_JUMP_WGT = 1
MD_JUMP_WGT = -1
LG_JUMP_WGT = -8
ALTING_WGT = -5
CONTOUR_WGT = 2

# --- HARMONIC FITNESS WEIGHTS -----------------
UNISON_WGT = -15
THIRD_WGT = 10
SIXTH_WGT = 10
FOURTH_WGT = 5
FIFTH_WGT = 6
SEVENTH_WGT = 3
SECOND_WGT = -15
TRITONE_WGT = -10

# ***************************************************************
# ----------------------- MUSIC GENOME CLASS ------------------------------
# ***************************************************************

# defs and classe for the musical "genome"
class musicGenome:
    def __init__(self):
        # 16 random values for 16 random notes
        # for each voice as well!
        self.melody = np.random.rand(TOTAL_NOTES)
        self.upper = np.random.rand(TOTAL_NOTES)
        self.middle = np.random.rand(TOTAL_NOTES)
        self.bass = np.random.rand(TOTAL_NOTES)

        # fitness score for how "good" it sounds
        self.fitness = 0.0

    # this mutates a couple of the notes to try out modified musical lines
    def mutate(self):
        # !***CHANGE THIS NUMBER TO CHANGE NUMBER OF NOTES MUTATED***!
        # right, now it's set at mutating 1-3 notes.

        # randomly pick a voice line to mutate
        voiceName = random.choice(['melody', 'upper', 'middle', 'bass'])

        # get the gene array for this voice
        voice = getattr(self, voiceName)

        # randomly pick 1-3 notes to mutate
        numMut = random.randint(*MUT_AMOUNT)
        noteIndex = random.sample(range(TOTAL_NOTES), numMut)

        # re-randomize the notes we want to mutate
        for note in noteIndex:
            voice[note] = random.random()

    # Note: ***
    # MAY NOT NEED DUPLICATE FUNCTION MIGHT TAKE OUT LATER!!!!!
    def copy(self):
        copy = musicGenome()
        copy.melody = self.melody.copy()
        copy.upper = self.upper.copy()
        copy.middle = self.middle.copy()
        copy.bass = self.bass.copy()
        copy.fitness = self.fitness
        return copy
    
# ***************************************************************
# ----------------------- GENOME TO MIDI ------------------------------
# ***************************************************************

# def to convert the musicGenome to raw MIDI
def genomeToMIDI(genome, voiceName):
    # global vars now take care of this setup***

    # make vars for the numbers we just set.
    minRange, maxRange = VOICES_RANGE[voiceName]
    endNote = END_NOTES[voiceName]

    # we can't modify the original musicGenome,
    # so we need to copy it then alter that one
    genomeCopy = genome.copy()

    # unless you force these lines to resolve they sound like garbage,
    # so here you force the last index to be the endnote found earlier
    # based on the voice
    if minRange <= endNote <= maxRange:
        genomeCopy[TOTAL_NOTES-1] = (endNote - minRange)/(maxRange - minRange)

    # now we need to put all this into an actual MIDI format
    midi = []

    # map each note or index in the music genome copy
    # to midi directly, and make it be the scale specced above^^
    for gene in genomeCopy:

        # map the random music genome individual gene/note
        # to a MIDI note, the calculation:
        # rawMIDI = 67 + (gene * 5) = 69.5 round down for int to 69 -> A4
        rawMIDI = int(minRange + (gene * (maxRange-minRange)))

        # apply the key to the midi notes now that we setup earlier
        # C scale here for now!!! ***
        scaleMod = rawMIDI%12 # 12 cause keys on the piano

        # based on the midi note now, we need to find the closest
        # note that exists in the scale (C scale for now!!!)
        closestNote = min(SCALE_VALS, key=lambda x: abs(x - scaleMod))

        # now we have the modifier and the closest note,
        # so apply the swap to the closest note based on these
        # these things
        scaleApply = rawMIDI + (closestNote - scaleMod)

        # *** additional safety check here!
        # i was noticing sometimes it would get funky if I didnt
        # have this safety measure to make sure it doesnt go to a note
        # thats out of the range for the voice!
        checkedNote = max(minRange, min(maxRange, scaleApply))

        # now just add the checked note to the midi list
        midi.append(checkedNote)

    # return the midi as a np array of ints
    return np.array(midi, dtype=int)

# ***************************************************************
# ----------------------- GENOME TO MUSIC21 ------------------------------
# ***************************************************************

# def for func to translate from a music Genome to the music21 format
def genomeToMus21(genome, voiceName):
    # music21 empty music staff, time sig, and key
    music = stream.Part()
    music.append(meter.TimeSignature(TIME_SIG))
    music.append(key.Key(KEY))
    music.append(tempo.MetronomeMark(number=TEMPO))

    # translate from the genome to MIDI
    midi = genomeToMIDI(genome, voiceName)

    # int to track measure num
    m = 0

    # build 4 measures with 4 quarter notes each
    for i in range(MEASURES):
        measure = stream.Measure(number=i+1)

        # now go beat by beat (4)
        for beat in range(BEATS_PER_MEASURE):
            # now create quarter notes with the pitch
            # of the MIDI array
            n = note.Note(pitch.Pitch(midi=midi[m]), quarterLength=1.0)
            measure.append(n)
            m+=1

        # append the measure now that the beats are built
        music.append(measure)

    # return it in its final format finally.
    return music

# ***************************************************************
# ----------------------- HELPER FUCNTION FOR ALTERNATING CHECKS ------------------------------
# ***************************************************************
# i wanted this to be super dynamic and adaptable depending on how long you set
# the measures and overall composition to be
def alternatingNotesChecker(voiceLine):
    # intial var to keep track of alternating notes penalty overall
    altPenalty = 0

    # go through each pattern of 2,3,4,5 or ABAB, ABCABC, ABCDABCD, etc.
    for patLength in range(2, min(ALT_MAX_CHECK+1, TOTAL_NOTES // 2+1)):

        # calulate a dynamic weight penalty based on how large the pattern is
        # bigger patterns have less impact than the smaller ones like ABAB
        wgt = 1.0/(patLength-1)

        # check each index for this pattern length where this
        # pattern could feasibly happen
        for i in range(TOTAL_NOTES - patLength*2 + 1):

            # get the possible pattern and the possible repeat at the
            # index we're at now
            firstPattern = tuple(voiceLine[i : i+patLength])
            matchPattern = tuple(voiceLine[i+patLength : i+patLength * 2])

            # now we can actually check if these two patterns match AND that
            # the pattern uses the exact number of pattern length notes
            if firstPattern == matchPattern and len(set(firstPattern)) == patLength:
                altPenalty += wgt

    return altPenalty

# ***************************************************************
# ----------------------- FITNESS FOR FULL COMPOSITION ------------------------------
# ***************************************************************

# fitness evaluation function for all 4 lines at once.
# this is a cohesive unified approach as opposed to a GA after each line
# is initial generated. This simplifies the process, and makes it so you can 
# do this not only faster but so every line actually adjusts instead of just
# the ones generating.
def fullCompFitness(genome):

    # first thing is to convert all the genome lines to MIDI lines
    melodyMIDI = genomeToMIDI(genome.melody, 'melody')
    upperMIDI = genomeToMIDI(genome.upper, 'upper')
    middleMIDI = genomeToMIDI(genome.middle, 'middle')
    bassMIDI = genomeToMIDI(genome.bass, 'bass')

    # set fitness of 0 to start out with
    fitness = 0.0

    # now we evaluate the fitness for each voice line
    # this section is for the horizontal checks only:
    # (note variety, jump/movement smoothness, anti-2 note alt check, and contour check)
    for voiceLine in [melodyMIDI, upperMIDI, middleMIDI, bassMIDI]:

        # VARIETY::
        # Im adding this cause I want the versions with more unique notes
        # be favored heavier, as it should be more "interesting"
        uniqueness = len(set(voiceLine))
        fitness += uniqueness * UNIQUE_WGT

        # SMOOTHNESS::
        # I want to reward the fitness score if there is more movement
        # THIS IS THE ONE IM THE MOST IFFY ABOUT
        # Its hard cause you dont want to move too much cause it sounds like trash,
        # but also want some movement so it doesnt just play the same note over
        # and over again
        moveNum = 0
        for i in range(1,TOTAL_NOTES):
            # see what the move from this note from the prev note looks like
            noteInterval = abs(voiceLine[i] - voiceLine[i-1])

            # ive got a check here for the movements I've thought
            # of, but these numbers seem ok? But im still not 
            # 100% on all of these.
            if noteInterval == 0:
                fitness -= SAME_NOTE_WGT # same note
            elif noteInterval <= 2:
                fitness += STEPWISE_WGT # step motion. Im a fan of this based on what ive seen
                moveNum += 1
            elif noteInterval <= 4:
                fitness += SM_JUMP_WGT # smaller jump
                moveNum += 1
            elif noteInterval <= 7:
                fitness += MD_JUMP_WGT # medium size-ish leap
                moveNum += 1
            else:
                fitness += LG_JUMP_WGT # XL/jumbo leap
                moveNum += 1
            # from what I saw in my earlier testing, if you didnt restrict
            # the bigger leaps it would sound like garbage for every line
            # pretty consistently. I'm not confident these numbers are perfect,
            # but they seemed pretty good to me, so I stuck with them for now.
    
        # Note****:
        # maybe penalize if it moves too much?? I cant decide if I like this
        # or not when I was testing
        # if moveNum > TOTAL_NOTES-3:
        #   fitness -= 3 * (moveNum - (TOTAL_NOTES-3))

        # ANTI-A-B-A-B:
        # in the earlier iterations i had, it would very consistently 
        # get stuck in a A-B-A-B pattern, and I think it def still can,
        # but its not as common as it used to be which was the main issue.
        # if it happens sometimes whatever, but in old versions it was almost
        # every part of every line, which was pretty boring.
        altPenalty = alternatingNotesChecker(voiceLine)
        fitness += altPenalty * ALTING_WGT
            
        # CONTOUR:
        # im a sucker for changes in the overall direction or the contour
        # of a line, so I wanted to make sure I added that here in the initial
        # GA construction for the lines, as I think it makes it way more interesting
        dirChangeNum = 0
        for i in range(1, TOTAL_NOTES-1):
            # im basically checking if it goes up, then after that it goes down
            goUp = voiceLine[i] - voiceLine[i-1]
            afterGoDown = voiceLine[i+1] - voiceLine[i]
            if goUp * afterGoDown < 0:
                dirChangeNum += 1
        fitness += dirChangeNum * CONTOUR_WGT

    # Note****:
    # I noticed that doing harmonics AFTER all the horizontal checks
    # helped a ton with how it sounds overall, so keep these seperate and not
    # in the same voice loop. Horizontal -> harmonics for melody only -> PSO for chords & verticality.
    # if you have any other checks for the GA side of things, just put them in another
    # seperate voice loop.
    # --------------------
    # HARMONIC CHECKING::::
    # This is the part I'm least comfortable with so Josef I would love some help here
    # right im literally just checking raw relations and giving fitness boosts/decrements
    # based ONLY on how each OTHER LINE is related to JUST THE MELODY HARMONICALLY,
    # in order to have some semblance of sounding decent before I do the PSO verticality check.
    # This could almost definetley be improved but I'm not enough into theory to know how lol
    for voiceLine in [upperMIDI, middleMIDI, bassMIDI]:
        # for each note
        for i in range(TOTAL_NOTES):
            noteInterval = abs(voiceLine[i] - melodyMIDI[i])
            # pretty easy to find how its related just by a mod 12
            noteIntervalClassif = noteInterval % 12

            # especially reward things that sound nice or promote consonances,
            # then decrement fitness for things that cause dissonance.
            # other new relations can get pushed once the chords come around.
            if noteIntervalClassif == 0:
                fitness += UNISON_WGT          # boring and lame cause same note smh
            elif noteIntervalClassif in [3,4]: # this covers thirds
                fitness += THIRD_WGT
            elif noteIntervalClassif in [8,9]: # sixths
                fitness += SIXTH_WGT
            elif noteIntervalClassif == 5:     # fourth
                fitness += FOURTH_WGT
            elif noteIntervalClassif == 7:     # fifth
                fitness += FIFTH_WGT
            elif noteIntervalClassif in [10,11]: # seventsh
                fitness += SEVENTH_WGT
            elif noteIntervalClassif in [1,2]: # seconds
                fitness += SECOND_WGT
            elif noteIntervalClassif == 6:     # tritone
                fitness += TRITONE_WGT

    # Note::: ********************
    # Vertically checks used to go here in my old iterations, but I've since
    # moved that to the 2nd pass through on the PSO run through. ***********

    return fitness

# Note::: ************
# IM THINKING ABOUT MAKING THIS A TOURNAMENT SELECTION MAYBE!???????

# ***************************************************************
# ----------------------- GA GENOME EVOLOUTION ------------------------------
# ***************************************************************

# function for the actual GA evolution for the voice lines
def evolveMusicGenome():
    # here we'll start with the intial random musicGenome then
    # just keep iterating on it with the earlier fitness and evolution functions
    # until we hit the number of generations specced above
    # default: 1000

    GAgenome = musicGenome()
    GAgenome.fitness = fullCompFitness(GAgenome)

    # now we need to loop through the evoloution for all the generations
    for gen in range(GENERATIONS):
        # first step is to make a copy of the OG genome,
        # then mutate the copy!
        child = GAgenome.copy()
        child.mutate()

        # run the fitness evaluation on the offspring of the genome
        child.fitness = fullCompFitness(child)

        # check if its better, if it is, keep it.
        if child.fitness > GAgenome.fitness:
            GAgenome = child

    # I need to see what the best fitness score is getting to at the end so print it
    # print(f" FINAL BEST FITNESS SCORE: {GAgenome.fitness:.1f}")
    return GAgenome


# ***************************************************************
# ----------------------- PSO CHORD STUFF ------------------------------
# ***************************************************************
# ...

# ***************************************************************
# ----------------------- MAIN LOOP ------------------------------
# ***************************************************************
# heres the main loop for the overall genome generation -> evoloution ->
# -> then to music21 and MIDI files

# run the GA evolution loop
evolvedComp = evolveMusicGenome()

# now translate the MIDI to music21 format
melodyOnly = genomeToMus21(evolvedComp.melody, 'melody')
upperOnly = genomeToMus21(evolvedComp.upper, 'upper')
middleOnly = genomeToMus21(evolvedComp.middle, 'middle')
bassOnly = genomeToMus21(evolvedComp.bass, 'bass')

# I want to show how it evolves over time
# Note for whoever is reading this:
# There are already papers that go over the fact that we can totally generate
# random music that sounds "ehhh" based off purely a GA with minimal restraints,
# and over a chromatic scale. However, ours is different as we initially restrict
# range to be suitable and possible for piano chords, as well as restricting the key 
# to C in order for it to actually sound mostly listenable. Without the fitness scores
# though it sounds like even more total garbage, so our GA does something, but its
# not as all powerful because our next part is the truly unique part. We can use a PSO
# algorithm to actually figure out what chords and relational movements and sounds
# will sound the "best". Our goal is not just to generate random music that sounds "ehhhh",
# its to show we can generate music that sounds decent with the right constraints with a GA,
# and then go through a 2nd pass with a PSO to actually build nice chords and relational sounding music,
# which from what I'm seeing is a totally unique and untapped idea. I haven't seen a paper covering
# this during my search.
# -------
# anways, I want a MIDI file with just the melody:
melodyOnly.write('midi', 'melody.mid')

# one with just the upper and melody:
upperAndMelody = stream.Score()
upperAndMelody.append(melodyOnly)
upperAndMelody.append(upperOnly)
upperAndMelody.write('midi', 'upper.mid')

# one with just the middle and melody:
middleAndMelody = stream.Score()
middleAndMelody.append(melodyOnly)
middleAndMelody.append(middleOnly)
middleAndMelody.write('midi', 'middle.mid')

# one with just the bass and melody:
bassAndMelody = stream.Score()
bassAndMelody.append(melodyOnly)
bassAndMelody.append(bassOnly)
bassAndMelody.write('midi', 'bass.mid')

# and finally with all four together BEFORE THE PSO:
fullComp = stream.Score()
fullComp.insert(0, instrument.Piano())
fullComp.append(melodyOnly)
fullComp.append(upperOnly)
fullComp.append(middleOnly)
fullComp.append(bassOnly)
fullComp.write('midi', 'full.mid')

# run the PSO evolution chord loop
# ...
# this broke so I'll add this back later.



    
