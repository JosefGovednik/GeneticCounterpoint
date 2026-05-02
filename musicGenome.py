from music21 import *
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ***************************************************************
# ----------------------- GLOBAL VARS ------------------------------
# ***************************************************************
# new global config variables so you can change the key, scale,
# tempo, time sig, etc.

# num of generations and number of notes to be mutated every evolution step
GENERATIONS = 1000
MUT_AMOUNT = (1, 4)   # low,high of possible notes to be mutated every iteration

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

# these are the chord definitions for what im checking later in the PSO
DIATONIC_CHORDS = {
    'I':   (0, 4, 7),    # C major   (C-E-G)
    'ii':  (2, 5, 9),    # D minor   (D-F-A)
    'iii': (4, 7, 11),   # E minor   (E-G-B)
    'IV':  (5, 9, 0),    # F major   (F-A-C)
    'V':   (7, 11, 2),   # G major   (G-B-D)
    'vi':  (9, 0, 4),    # A minor   (A-C-E)
    'vii': (11, 2, 5),   # B dim     (B-D-F)
}

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
INDV_SPAN = 8

# added new bass span value! chord progs were getting
# stuck due to bass centering, so I made it so bass could move more in a given range
BASS_SPAN = 11

# this will then auto calculate the range of each voice based on the center and indv spans.
VOICES_RANGE = {
    'melody': (VOICE_CENTER+1, VOICE_CENTER+1 + INDV_SPAN),
    'upper': (VOICE_CENTER-2, VOICE_CENTER-2 + INDV_SPAN),
    'middle': (VOICE_CENTER-4, VOICE_CENTER-4 + INDV_SPAN),
    'bass': (VOICE_CENTER-6, VOICE_CENTER-6 + BASS_SPAN),
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
SAME_NOTE_WGT = 0
UNIQUE_WGT = 4
STEPWISE_WGT = 4
SM_JUMP_WGT = 2
MD_JUMP_WGT = 0
LG_JUMP_WGT = -3
ALTING_WGT = -5
CONTOUR_WGT = 2

# --- HARMONIC FITNESS WEIGHTS -----------------
UNISON_WGT = -15
THIRD_WGT = 12
SIXTH_WGT = 10
FOURTH_WGT = 6
FIFTH_WGT = 8
SEVENTH_WGT = 3
SECOND_WGT = -5
TRITONE_WGT = -10

# ***************************************************************
# ----------------------- 2-STAGE PSO PARAMS ------------------------------
# ***************************************************************
# stage 1 vars
PSO_S1_PARTICLES = 50
PSO_S1_ITERATIONS = 500
PSO_S1_MAX_MOVE = 3 # +/- this many semitones up or down

# stage 2 vars
PSO_S2_PARTICLES = 50
PSO_S2_ITERATIONS = 500

# intertia settings for PSO
# ideally you want to go high to low to encourage localization
# post-exploration
PSO_INERTIA_START = 0.9
PSO_INERTIA_END = 0.4

# chord weights
CHORD_UNIQUE_WGT = 20  # this is to reward 3 or 4 unique notes in the chord to promote variation and not the same note
CHORD_THIN_WGT = -10
CHORD_THIRD_WGT = 18
CHORD_FIFTH_WGT = 15
CHORD_SIXTH_WGT = 15
CHORD_OCTAVE_WGT = 10
CHORD_CLUSTER_WGT = -50
CHORD_TRITONE_WGT = -35

# voice leading weights and params
SMOOTH_LEADING_WGT = 15
LEAP_LEADING_WGT = -25
MEASURE_LEAP_LEAP_WGT = -40

# voice leading params for PSO
# - Without this, it didnt really have a great way of figuring out
# what was good across multiple measures, instead of just neighboring ones
# for 2, 3, and 4 measures respectively
PROG_QUALITY_2 = 150
PROG_QUALITY_3 = 250
PROG_QUALITY_4 = 350

# measure-local 2 chord prog weights
P_1_5_WGT = 40
P_5_1_WGT = 50
P_4_1_WGT = 35
P_1_4_WGT = 30
P_2_5_WGT = 45
P_5_4_WGT = 30
P_6_4_WGT = 25
P_4_5_WGT = 35


# measure-local 3 chord prog weights
P_1_5_1_WGT = 80
P_1_4_5_WGT = 75
P_2_5_1_WGT = 85
P_1_6_4_WGT = 70
P_6_4_5_WGT = 72


# measure-local 4 chord prog weights
P_1_5_6_4_WGT = 120
P_6_4_1_5_WGT = 115
P_1_6_4_5_WGT = 110
P_1_4_5_1_WGT = 120


# 2 chord prog weights
prog2_map = {
    ('I', 'V'): P_1_5_WGT,
    ('V', 'I'): P_5_1_WGT,
    ('IV', 'I'): P_4_1_WGT,
    ('I', 'IV'): P_1_4_WGT,
    ('ii', 'V'): P_2_5_WGT,
    ('V', 'IV'): P_5_4_WGT,
     ('vi', 'IV'): P_6_4_WGT,
    ('IV', 'V'): P_4_5_WGT,
}

# 3 chord prog weights
prog3_map = {
    ('I', 'V', 'I'): P_1_5_1_WGT,
    ('I', 'IV', 'V'): P_1_4_5_WGT,
    ('ii', 'V', 'I'): P_2_5_1_WGT,
    ('I', 'vi', 'IV'): P_1_6_4_WGT,
    ('vi', 'IV', 'V'): P_6_4_5_WGT,
}

# 4 chord prog weights
prog4_map = {
    ('I', 'V', 'vi', 'IV'): P_1_5_6_4_WGT,
    ('vi', 'IV', 'I', 'V'): P_6_4_1_5_WGT,
    ('I', 'vi', 'IV', 'V'): P_1_6_4_5_WGT,
    ('I', 'IV', 'V', 'I'): P_1_4_5_1_WGT,
}

# id chord global vars
# again Josed if you think any of these values should be changed when
# you test or see this fr change any and all of them. Literally just did this
# by listening and im not confident in these vals lol. They did sound pretty decent
# from my testing though.
ID_CHORDS_WGT = 3
ID_NONCHORDS_WGT = -2
ID_BASS_ROOT_WGT = 4
ID_BASS_INV_WGT = 1
ID_BASS_OFF_WGT = -2
ID_TRIAD_WGT = 2
ID_MIN_CONF = 2


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
# ----------------------- Stage 1&2 PSO CLASSES ------------------------------
# ***************************************************************
# I have a seperate particle class for stage 1 and stage 2,
# as both have different goals and do totally different things
class stage1Particle:
    def __init__(self):
        # give each line a range where the note can move in semitones,
        # and also make sure it stays in the scale we're in so 
        # nothing completely insane happens
        self.pos = {
            'melody': np.random.uniform(-PSO_S1_MAX_MOVE, PSO_S1_MAX_MOVE, TOTAL_NOTES),
            'upper': np.random.uniform(-PSO_S1_MAX_MOVE, PSO_S1_MAX_MOVE, TOTAL_NOTES),
            'middle': np.random.uniform(-PSO_S1_MAX_MOVE, PSO_S1_MAX_MOVE, TOTAL_NOTES),
            'bass': np.random.uniform(-PSO_S1_MAX_MOVE, PSO_S1_MAX_MOVE, TOTAL_NOTES),
        }

        # velocity param, which changes how fast each line ends
        # up changing which works with the interia val as well.
        self.velocity = {
            'melody': np.random.uniform(-1, 1, TOTAL_NOTES),
            'upper': np.random.uniform(-1, 1, TOTAL_NOTES),
            'middle': np.random.uniform(-1, 1, TOTAL_NOTES),
            'bass': np.random.uniform(-1, 1, TOTAL_NOTES),
        }

        # personal bests, or the personal best position and fit
        # that this ind particle has found so far
        self.pBestPos = {k: v.copy() for k, v in self.pos.items()}
        self.pBestFit = -float('inf')
        self.fitness = 0.0

class stage2Particle:
    def __init__(self):
        # start with a random rearrangment of the measures to begin
        self.order = list(range(MEASURES))
        random.shuffle(self.order)

        # pb: personal best order and fit that this ind particle has found
        self.pBestOrder = self.order.copy()
        self.pBestFit = -float('inf')
        self.fitness = 0.0

    # method to swap two measures with each other
    # this is the real meat and potatoes of stage 2 in tandem with 
    # the next method
    def swap(self):
        i, j = random.sample(range(MEASURES), 2)
        self.order[i], self.order[j] = self.order[j], self.order[i]

    # another huge piece, swap stuff further to actually promote
    # good overall progressions across measures, and this is how the particles,
    # actually follow their own or global best
    def moveBetter(self, idealOrder, prob):
        for i in range(MEASURES):
            # basically just its not the ideal order, and we're in
            # the probability thrreshold
            if self.order[i] != idealOrder[i] and random.random() < prob:
                # find where the ideal value is for the current given order
                j = self.order.index(idealOrder[i])
                # then swap the order to match the ideal order for this position
                self.order[i], self.order[j] = self.order[j], self.order[i]

# ***************************************************************
# ----------------------- SCALE SNAP HELP FUNC ------------------------------
# ***************************************************************
# scale was consistently out of range, so instead of trying to make
# it bulletproof from the start, I'm adding a check to the end to make sure it
# actually snaps to whatever scale you have set.
def scaleSnapper(rawNote, minRange, maxRange):
    # build the temp valid scale range notes first
    candidates = []
    for notes in range(int(minRange), int(maxRange)+1):
        if (notes%12) in SCALE_VALS:
            candidates.append(notes)

    # for some reasone if this breaks we can return the closest notes within reason
    if not candidates:
        return max(minRange, min(maxRange, rawNote))
    
    # but if we're good then return all the nearest notes in the scale
    # that we "snap" to
    return min(candidates, key=lambda x: abs(x-rawNote))


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
        rawMIDI = int(round(minRange + (gene * (maxRange-minRange))))

        # fasttrack for scale snapping, as we can use our helper function
        # to just snap all the notes here to the closest based on what scale is
        checkedNote = scaleSnapper(rawMIDI, minRange, maxRange)

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
# ----------------------- MIDI TO GENOME CONVERTER------------------------------
# ***************************************************************
# im basically making this function in case I need it later, or have a use case
# for it in the future
# New: This actually helps with the PSO stuff, and can be used
# to put it back from midi to the genome so we can see whats going on in it.
def midiToGenome(midiComp):
    translatedGenome = musicGenome()

    for voice in ['melody', 'upper', 'middle', 'bass']:
        midi = midiComp[voice]
        minRange, maxRange = VOICES_RANGE[voice]

        # move each midi value transalated back to a genome val 0-1
        genes = []
        for note in midi:
            gene = (note-minRange) / (maxRange-minRange)
            gene = max(0.0, min(1.0, gene))
            genes.append(gene)

        # set it to the music genome
        setattr(translatedGenome, voice, np.array(genes))
        
    # and return it
    return translatedGenome


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
                fitness += SAME_NOTE_WGT # same note
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
            elif noteIntervalClassif in [10,11]: # sprogsh
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
# ----------------------- PSO FITNESS FUNCS ------------------------------
# ***************************************************************
# this func checks the leading voice quality so we can actually see
# how well notes connect overall, especially over the span of measures
def leadVoiceEval(midiComp):
    fitness = 0.0
    voices = [
        midiComp['melody'],
        midiComp['upper'],
        midiComp['middle'],
        midiComp['bass'],
    ]

    # for each voice, reward different types of motion from notes
    # and from measure to measure to penalize huge jumps that
    # feel abrupt and weird
    for voice in voices:
        for i in range(1, TOTAL_NOTES):
            # get the interval from voice to voice
            interval = abs(voice[i] - voice[i-1])

            # we like smooth step-wsie motion
            if interval <= 2:
                fitness += SMOOTH_LEADING_WGT
            # we do not however like big jumps
            elif interval >= 7:
                fitness += LEAP_LEADING_WGT

            # and we ESPECIALLY dont like big jumps
            # from measure to measure. This is what I found during testing
            # sounded reallllly bad. Measure cohesion drastically improved
            # the PSO double pass sound quality
            if(i % BEATS_PER_MEASURE == 0) and interval >= 5:
                fitness += MEASURE_LEAP_LEAP_WGT
    return fitness

# this is for the stage 1 PSO pass, and it looks at
# the veritcal chord quality, and will shift things around 
# in a globally defined local range. This is the local swarm part basically.
def stage1ChordOptimize(midiComp):
    fitness = 0.0

    melodyNotes = midiComp['melody']
    upperNotes = midiComp['upper']
    middleNotes = midiComp['middle']
    bassNotes = midiComp['bass']

    # go through every individual beat and look at it like a chord
    for i in range(TOTAL_NOTES):
        chord = [melodyNotes[i], upperNotes[i], middleNotes[i], bassNotes[i]]

        # look at how "full" the chord is, or how many unique notes it has
        # *** I'm still a little iffy on this one, but I thought I'd
        # at least include it to promote diversity on the initial PSO pass
        uniqueNotes = len(set(chord))
        if uniqueNotes >= 3:
            fitness += CHORD_UNIQUE_WGT
        elif uniqueNotes == 2:
            fitness += CHORD_THIN_WGT

        # now check the chord intervals to further refine the process
        # from the GA but with stricter constraints, inertia, and an added
        # locality attribute now
        for j in range(len(chord)):
            for k in range(j+1, len(chord)):
                interval = abs(chord[j] - chord[k])
                intervalType = interval%12

                # interval types with associated fitness vals
                # these vals are all just placeholders that sounded fine
                # enough to me, but if you find better vals,
                # just swap them in the global vars
                if intervalType in [3, 4]:      #maj/min thirds
                    fitness += CHORD_THIRD_WGT
                elif intervalType in [8, 9]:    #maj/min sixths
                    fitness += CHORD_SIXTH_WGT 
                elif intervalType == 7:         #perfect fifth
                    fitness += CHORD_FIFTH_WGT 
                elif intervalType == 0 and interval == 12:  #octave
                    fitness += CHORD_OCTAVE_WGT 
                elif intervalType in [1, 2]:    #maj/min seconds
                    fitness += CHORD_CLUSTER_WGT 
                elif intervalType == 6:         #tritone
                    fitness += CHORD_TRITONE_WGT 
    
    # now we have all the info for the chord quality, we 
    # need to evaluate the leading voices for further quality
    leadingVoiceFitBonus = leadVoiceEval(midiComp)
    fitness += leadingVoiceFitBonus
    return fitness

# because we're looking at basing a lot of this PSO double pass
# system on the chord quality of all the voices combined, this func
# that evaluates unifrom chord type is crucial. 
def idChord(notes):
    # change whatever note is to a pitch class, so it works
    # for any octave
    pitch = [int(n)%12 for n in notes]
    pitchSet = set(pitch)
    bassPitch = int(notes[3])%12
    bestChord = 'Unknown'
    bestScore = -float('inf')

    # try all the different diatonic chords setup globally and see how 
    # well each one matches
    for chordType, (rootPitch, thirdPitch, fifthPitch) in DIATONIC_CHORDS.items():
        chordSet = {rootPitch, thirdPitch, fifthPitch}

        # see how many of the four voices land on chord tones vs. 
        # non chord tones to see where we're at as of now
        chords = len(pitchSet & chordSet)
        nonChords = len(pitchSet - chordSet)

        # setup a base score, where chords present are wieghted heavier
        # and positively while the opposite is true for non chords
        chordScore = chords*ID_CHORDS_WGT + nonChords*ID_NONCHORDS_WGT

        # the bass usually gives a super solid idea of what the root
        # should be for a chord, so go based off that usually.
        if bassPitch == rootPitch:
            chordScore += ID_BASS_ROOT_WGT
        elif bassPitch == thirdPitch or bassPitch == fifthPitch:
            chordScore += ID_BASS_INV_WGT
        else:
            chordScore += ID_BASS_OFF_WGT

        # extra points if we form a complete triad!
        if chords == 3:
            chordScore += ID_TRIAD_WGT

        # now check our best, and if our local chord score is better,
        # then replace it!
        if chordScore > bestScore:
            bestScore = chordScore
            bestChord = chordType

    # if the chord score isnt high enough for the best, then we keep it as unknown
    # and move on
    if bestScore < ID_MIN_CONF:
        return 'Unknown'
    
    # now return whatever our new (or old) best chord is
    return bestChord

# NOW, we have the main stage 2 PSO driver, which is where it looks at how
# the progression is on a beat by beat basis and on a measure by measure basis
# by looking at if the progression spans across measures, as that means the stage 2
# bread and butter, measure reordering, actually matters in the end.
def progAnalyzer(midiComp):
    melody = midiComp['melody']
    upper = midiComp['upper']
    middle = midiComp['middle']
    bass = midiComp['bass']
    fitness = 0.0

    # now we need to build a list of chord types but using ALL
    # of the voices so we ca evaluate chord quality across
    # multiple voices, not just bass like in my previous version
    chordTypes = []
    for i in range(TOTAL_NOTES):
        # get the chord across all voices and append it to the chord types
        chord = [melody[i], upper[i], middle[i], bass[i]]
        chordTypes.append(idChord(chord))

    # now we can kinda "collapse" neighboring chords into a singular event
    # so instead of looking at two seperate chords as a progression instance,
    # we have on progression instance that contains two chords.
    progs = []
    for i, c in enumerate(chordTypes):
        if not progs or progs[-1][0] != c:
            progs.append((c, i))

    # now we have a helper func def to see if the chord prog instance
    # we just made actually spans across a measure
    def progAcrossMeasures(progInstance):
        start = progInstance[0][1]
        end = progInstance[-1][1]
        startMeasure = start // BEATS_PER_MEASURE
        endMeasure = end // BEATS_PER_MEASURE
        return startMeasure != endMeasure
    
    # 2 chord span evaluator part
    # cant be last note plus another so minus one
    for i in range(len(progs)-1):
        # get the prog instance window
        progWindow = progs[i:i+2]

        # get the prog from the window now based on how many chords were looking at
        prog = (progWindow[0][0], progWindow[1][0])
        
        # now see if we apply a bonus for a prog across measures too!
        spanBonus = prog2_map.get(prog, 0)
        if spanBonus and progAcrossMeasures(progWindow):
            spanBonus += PROG_QUALITY_2
        fitness += spanBonus

    # 3 chord span evaluator part
    # cant be last note plus another so minus two
    for i in range(len(progs)-2):
        # get the prog instance window
        progWindow = progs[i:i+3]

        # get the prog from the window now based on how many chords were looking at
        prog = (progWindow[0][0], progWindow[1][0], progWindow[2][0])
        
        # now see if we apply a bonus for a prog across measures too!
        spanBonus = prog3_map.get(prog, 0)
        if spanBonus and progAcrossMeasures(progWindow):
            spanBonus += PROG_QUALITY_3
        fitness += spanBonus


    # 4 chord span evaluator part
    # cant be last note plus another so minus three
    for i in range(len(progs)-3):
        # get the prog instance window
        progWindow = progs[i:i+4]

        # get the prog from the window now based on how many chords were looking at
        prog = (progWindow[0][0], progWindow[1][0], progWindow[2][0], progWindow[3][0])
        
        # now see if we apply a bonus for a prog across measures too!
        spanBonus = prog4_map.get(prog, 0)
        if spanBonus and progAcrossMeasures(progWindow):
            spanBonus += PROG_QUALITY_4
        fitness += spanBonus

    # finally return fitness at the end for this progression
    return fitness


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
# ----------------------- PSO CHORD APPLY STUFF ------------------------------
# ***************************************************************
# to make thigns simpler after I kept on breaking this code for some reason
# with the addition of the STAGE 2 PSO stuff added on, I made a func def that
# just auto calls all the stage 2 stuff which for some reason helped stuff not break anymore
# think I was just messing up part of the call somewhere or something.
def stage2FitnessEasyCall(midiComp):
    fitness = 0.0
    fitness += progAnalyzer(midiComp)
    fitness += leadVoiceEval(midiComp)
    return fitness

# heres a function to actually apply all the chord adjustments made from the stage 1
# PSO passthrough
def applyStage1(musicGenome, particle):
    stage1Result = {}

    # go through each voice
    for voice in ['melody', 'upper', 'middle', 'bass']:
        # get the midi for the music genome for this respective voice
        midi = genomeToMIDI(getattr(musicGenome, voice), voice)

        # round the particle adjustments to whatever the nearest int rounded
        # semitone is. I had a lot of weird behavior with this part so I applied a
        # lot of rounding here
        adjusted = particle.pos[voice]
        adjustedMidi = midi + np.round(adjusted).astype(int)

        # get the new ranges based on the current voice
        minRange, maxRange = VOICES_RANGE[voice]

        # snap each note that has been adjusted to the nearest in scle
        # note with the help of the earlier helper function
        finalMidi = []
        for note in adjustedMidi:
            finalMidi.append(scaleSnapper(note, minRange, maxRange))

        stage1Result[voice] = np.array(finalMidi, dtype=int)

    # ALSO, reagardless of what the chord optimizer found we need to resovle
    # to our preffered end note, so do that here based on whatever is set globally.
    for voice in ['melody', 'upper', 'middle', 'bass']:
        stage1Result[voice][-1] = END_NOTES[voice]

    # return the result of stage 1 finally
    return stage1Result

# now heres the function to apply stage 2 cleanly to reduce complexity later in
# the PSO main work loops
def applyStage2(musicGenome, particle):
    stage2Result = {}

    # we need to find whatever the last measure as thats the one that
    # had its last note changed to resolve to our ideal end notes set globally,
    # so keep this one where it is basically
    lastMeasure = MEASURES-1

    # now we have to filter the order of the particle to try out
    # a new ordering but def keep the last measure as the last one, 
    # so we still resolve to our favored end chord
    nonFinalOrder = [measure for measure in particle.order if measure != lastMeasure]
    finalOrder = nonFinalOrder+[lastMeasure]

    # now go through each voice
    for voice in ['melody', 'upper', 'middle', 'bass']:
        midi = genomeToMIDI(getattr(musicGenome, voice), voice)

        # rebuild the line over each measure in the new order we just made.
        reordered = []
        for measure in finalOrder:
            # get the starting beat of the measure
            start = measure * BEATS_PER_MEASURE

            # get the ending beat of the measure
            end = (measure+1) * BEATS_PER_MEASURE

            # now place it in the reordered measure
            reordered.extend(midi[start:end])

        # now we need to put all our stage2results in into the array for each voice
        stage2Result[voice] = np.array(reordered, dtype=int)

    # finally return the final stage 2 result
    return stage2Result

# ***************************************************************
# ----------------------- PSO STAGE !1! MAIN LOOP ------------------------------
# ***************************************************************
# alright, heres the part where we actually go through all of the PSO stage 1 funcs
# and actually optimize the chords and do the real PSO loop work.
def stage1PSOWork(musicGenome):
    # init our swarm var for the PSO work, and have it ahve the right
    # number of particles based on our global vars
    swarm = [stage1Particle() for _ in range(PSO_S1_PARTICLES)]

    # now we to store our global best positon and fitness as a global
    # snapshot of whats there. The actual content might change, so using 
    # a reference could absoloutely break this. I've never experienced that though...
    globalBestPos = None
    globalBestFit = -float('inf')

    # do the first evaluation pass so we can do the PSO stuff right
    for particle in swarm:
        # adjust the chords and get the fitness of the particle
        chordsAdjusted = applyStage1(musicGenome, particle)
        particle.fitness = stage1ChordOptimize(chordsAdjusted)

        # if our new particle fitness we found is better than that
        # particle's known best, change it
        if particle.fitness > particle.pBestFit:
            particle.pBestFit = particle.fitness
            particle.pBestPos = {k: v.copy() for k, v in particle.pos.items()}

        # if our new particle fitness we found is better than our
        # current global best, than change it to whatever that is
        if particle.fitness > globalBestFit:
            globalBestFit = particle.fitness
            globalBestPos = {k: v.copy() for k, v in particle.pos.items()}

    # Now that we've done our initial evaluation and setup, we 
    # can now do the main PSO work loop here!
    for iterartion in range(PSO_S1_ITERATIONS):
        # heres the adaptive inertia value, where it'll change from high to low,
        # which will encourage more exploration at the start to lower at the end.
        # using the fucntion below to find this inertia value. This helped a ton when
        # added this as oppossed to the initial versions
        inertia = PSO_INERTIA_START - ((PSO_INERTIA_START-PSO_INERTIA_END) * iterartion/PSO_S1_ITERATIONS)

        # now go through each particle in the swarm
        for particle in swarm:
            # and for each particle go through each voice
            for voice in ['melody', 'upper', 'middle', 'bass']:
                # make two totally random note arrays
                random1 = np.random.random(TOTAL_NOTES)
                random2 = np.random.random(TOTAL_NOTES)

                # then do a cognitive and social calculation for going towards the 
                # local and global best based on the random note arrays we just made
                cognitive = 2.0*random1 * (particle.pBestPos[voice] - particle.pos[voice])
                social = 2.0*random2 * (globalBestPos[voice] - particle.pos[voice])

                # see how quickly or with what velocity it moves towards these new positions
                # aka how many notes it actually moves towards the local or global bests
                particle.velocity[voice] = inertia * particle.velocity[voice] + cognitive + social
                particle.pos[voice] = particle.pos[voice] + particle.velocity[voice]

                # now we need to make sure whatever our adjustments were from in 
                # between the global/local bests, that it puts us in a legal
                # range within our limits and scale we want.
                particle.pos[voice] = np.clip(particle.pos[voice], -PSO_S1_MAX_MOVE, PSO_S1_MAX_MOVE)

            # now we need to reevaluate after moving things based on the PSO logic
            chordsAdjusted = applyStage1(musicGenome, particle)
            particle.fitness = stage1ChordOptimize(chordsAdjusted)

            # if our new particle fitness we found is better than that
            # particle's known best, change it
            if particle.fitness > particle.pBestFit:
                particle.pBestFit = particle.fitness
                particle.pBestPos = {k: v.copy() for k, v in particle.pos.items()}

            # if our new particle fitness we found is better than our
            # current global best, than change it to whatever that is
            if particle.fitness > globalBestFit:
                globalBestFit = particle.fitness
                globalBestPos = {k: v.copy() for k, v in particle.pos.items()}

    # now we look and see what our best particle was and what adjustments it made,
    # then return that particle as a musicGenome!
    bestParticle = stage1Particle()
    bestParticle.pos = globalBestPos
    finalStage1Composition = applyStage1(musicGenome, bestParticle)
    return midiToGenome(finalStage1Composition)


# ***************************************************************
# ----------------------- PSO STAGE !2! MAIN LOOP ------------------------------
# ***************************************************************
# alright, heres the part where we actually go through all of the PSO stage 2 funcs
# and actually "optimize" the measure order and do the real PSO stage 2 loop work.
def stage2PSOWork(musicGenome):
    # init our swarm var for the PSO work, and have it ahve the right
    # number of particles based on our global vars
    swarm = [stage2Particle() for _ in range(PSO_S2_PARTICLES)]

    # now we to store our global best positon and fitness as a global
    # snapshot of whats there. The actual content might change, so using 
    # a reference could absoloutely break this. I've never experienced that though...
    globalBestOrder = None
    globalBestFit = -float('inf')

    # do the first evaluation pass so we can do the PSO stuff right
    for particle in swarm:
        # adjust the measures and get the fitness of the particle
        measuresAdjusted = applyStage2(musicGenome, particle)
        particle.fitness = stage2FitnessEasyCall(measuresAdjusted)

        # if our new particle fitness we found is better than that
        # particle's known best, change it
        if particle.fitness > particle.pBestFit:
            particle.pBestFit = particle.fitness
            particle.pBestOrder = particle.order.copy()

        # if our new particle fitness we found is better than our
        # current global best, than change it to whatever that is
        if particle.fitness > globalBestFit:
            globalBestFit = particle.fitness
            globalBestOrder = particle.order.copy()

    # Now that we've done our initial evaluation and setup, we 
    # can now do the main PSO work loop here!
    for iterartion in range(PSO_S2_ITERATIONS):
        # heres the adaptive inertia value, where it'll change from high to low,
        # which will encourage more exploration at the start to lower at the end.
        # using the fucntion below to find this inertia value. This helped a ton when
        # added this as oppossed to the initial versions
        inertia = PSO_INERTIA_START - ((PSO_INERTIA_START-PSO_INERTIA_END) * iterartion/PSO_S2_ITERATIONS)

        # now go through each particle in the swarm
        for particle in swarm:
            # move toward the personal best for this particle
            particle.moveBetter(particle.pBestOrder, prob=(1-inertia) * 0.5)

            # move toward the global best for this particle (not as heavily weighted as personal best)
            particle.moveBetter(globalBestOrder, prob=(1-inertia) * 0.2)

            # get a random number with our inertia to determine how far it moves
            # based on the local/global bests
            if random.random() < inertia*0.3:
                particle.swap()

            # adjust the measures and get the fitness of the particle
            measuresAdjusted = applyStage2(musicGenome, particle)
            particle.fitness = stage2FitnessEasyCall(measuresAdjusted)

            # if our new particle fitness we found is better than that
            # particle's known best, change it
            if particle.fitness > particle.pBestFit:
                particle.pBestFit = particle.fitness
                particle.pBestOrder = particle.order.copy()

            # if our new particle fitness we found is better than our
            # current global best, than change it to whatever that is
            if particle.fitness > globalBestFit:
                globalBestFit = particle.fitness
                globalBestOrder = particle.order.copy()

    # now we look and see what our best particle was and what adjustments it made,
    # then return that particle as a musicGenome!
    bestParticle = stage2Particle()
    bestParticle.order = globalBestOrder
    finalStage2Composition = applyStage2(musicGenome, bestParticle)
    return midiToGenome(finalStage2Composition)


# ***************************************************************
# ----------------------- GRAPHS FOLDER + VOICE PLOTTER ------------------------------
# ***************************************************************

GRAPHS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Graphs')
MIDI_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MIDI')

VOICE_COLORS = {
    'melody': '#E63946',
    'upper':  '#2A9D8F',
    'middle': '#E9C46A',
    'bass':   '#457B9D',
}

def plotVoices(genome, title, savePath):
    voiceData = {
        'melody': genomeToMIDI(genome.melody, 'melody'),
        'upper':  genomeToMIDI(genome.upper,  'upper'),
        'middle': genomeToMIDI(genome.middle, 'middle'),
        'bass':   genomeToMIDI(genome.bass,   'bass'),
    }

    allNotes = np.concatenate(list(voiceData.values()))
    yMin = int(allNotes.min()) - 1
    yMax = int(allNotes.max()) + 1

    fig, ax = plt.subplots(figsize=(14, 5))

    for voiceName, midiNotes in voiceData.items():
        color = VOICE_COLORS[voiceName]
        for beat, midiNote in enumerate(midiNotes):
            rect = mpatches.FancyBboxPatch(
                (beat + 0.05, midiNote - 0.45),
                0.9, 0.9,
                boxstyle='round,pad=0.05',
                linewidth=0,
                facecolor=color,
                alpha=0.85,
            )
            ax.add_patch(rect)

    for beat in range(TOTAL_NOTES + 1):
        ax.axvline(beat, color='#cccccc', linewidth=0.4, zorder=0)

    for m in range(MEASURES + 1):
        ax.axvline(m * BEATS_PER_MEASURE, color='#888888', linewidth=1.0, zorder=1)

    ax.set_xlim(0, TOTAL_NOTES)
    ax.set_ylim(yMin, yMax + 1)
    ax.set_xlabel('Beat')
    ax.set_ylabel('MIDI Pitch')
    ax.set_title(title)

    pitchRange = range(yMin, yMax + 2)
    ax.set_yticks(list(pitchRange))
    ax.set_yticklabels([pitch.Pitch(midi=p).nameWithOctave for p in pitchRange], fontsize=7)

    ax.set_xticks([m * BEATS_PER_MEASURE for m in range(MEASURES + 1)])
    ax.set_xticklabels([f'M{m+1}' if m < MEASURES else '' for m in range(MEASURES + 1)])

    legend_patches = [mpatches.Patch(color=VOICE_COLORS[v], label=v.capitalize()) for v in VOICE_COLORS]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {savePath}")


# ***************************************************************
# ----------------------- GA WEIGHT EXPERIMENTS ------------------------------
# ***************************************************************
# Each config tweaks a different aspect of the horizontal / harmonic
# fitness weights relative to the baseline to see how the GA responds.

GA_EXPERIMENT_CONFIGS = {
    'Baseline': {
        'SAME_NOTE_WGT': 0,  'UNIQUE_WGT': 4,   'STEPWISE_WGT': 4,
        'SM_JUMP_WGT': 2,    'MD_JUMP_WGT': 0,  'LG_JUMP_WGT': -3,
        'ALTING_WGT': -5,    'CONTOUR_WGT': 2,
        'UNISON_WGT': -15,   'THIRD_WGT': 12,   'SIXTH_WGT': 10,
        'FOURTH_WGT': 6,     'FIFTH_WGT': 8,    'SEVENTH_WGT': 3,
        'SECOND_WGT': -5,    'TRITONE_WGT': -10,
    },
    'Smoothness-Heavy': {
        'SAME_NOTE_WGT': 0,  'UNIQUE_WGT': 4,   'STEPWISE_WGT': 8,
        'SM_JUMP_WGT': 0,    'MD_JUMP_WGT': -2, 'LG_JUMP_WGT': -8,
        'ALTING_WGT': -5,    'CONTOUR_WGT': 2,
        'UNISON_WGT': -15,   'THIRD_WGT': 12,   'SIXTH_WGT': 10,
        'FOURTH_WGT': 6,     'FIFTH_WGT': 8,    'SEVENTH_WGT': 3,
        'SECOND_WGT': -5,    'TRITONE_WGT': -10,
    },
    'Variety-Heavy': {
        'SAME_NOTE_WGT': -3, 'UNIQUE_WGT': 8,   'STEPWISE_WGT': 4,
        'SM_JUMP_WGT': 2,    'MD_JUMP_WGT': 1,  'LG_JUMP_WGT': -3,
        'ALTING_WGT': -5,    'CONTOUR_WGT': 2,
        'UNISON_WGT': -15,   'THIRD_WGT': 12,   'SIXTH_WGT': 10,
        'FOURTH_WGT': 6,     'FIFTH_WGT': 8,    'SEVENTH_WGT': 3,
        'SECOND_WGT': -5,    'TRITONE_WGT': -10,
    },
    'Contour-Heavy': {
        'SAME_NOTE_WGT': 0,  'UNIQUE_WGT': 4,   'STEPWISE_WGT': 4,
        'SM_JUMP_WGT': 2,    'MD_JUMP_WGT': 0,  'LG_JUMP_WGT': -3,
        'ALTING_WGT': -5,    'CONTOUR_WGT': 7,
        'UNISON_WGT': -15,   'THIRD_WGT': 12,   'SIXTH_WGT': 10,
        'FOURTH_WGT': 6,     'FIFTH_WGT': 8,    'SEVENTH_WGT': 3,
        'SECOND_WGT': -5,    'TRITONE_WGT': -10,
    },
    'Harmonic-Heavy': {
        'SAME_NOTE_WGT': 0,  'UNIQUE_WGT': 4,   'STEPWISE_WGT': 4,
        'SM_JUMP_WGT': 2,    'MD_JUMP_WGT': 0,  'LG_JUMP_WGT': -3,
        'ALTING_WGT': -5,    'CONTOUR_WGT': 2,
        'UNISON_WGT': -20,   'THIRD_WGT': 20,   'SIXTH_WGT': 18,
        'FOURTH_WGT': 10,    'FIFTH_WGT': 14,   'SEVENTH_WGT': 1,
        'SECOND_WGT': -10,   'TRITONE_WGT': -18,
    },
    'Anti-Repeat-Heavy': {
        'SAME_NOTE_WGT': -3, 'UNIQUE_WGT': 4,   'STEPWISE_WGT': 4,
        'SM_JUMP_WGT': 2,    'MD_JUMP_WGT': 0,  'LG_JUMP_WGT': -3,
        'ALTING_WGT': -14,   'CONTOUR_WGT': 2,
        'UNISON_WGT': -15,   'THIRD_WGT': 12,   'SIXTH_WGT': 10,
        'FOURTH_WGT': 6,     'FIFTH_WGT': 8,    'SEVENTH_WGT': 3,
        'SECOND_WGT': -5,    'TRITONE_WGT': -10,
    },
}

# how many GA generations to run per experiment config (kept short for speed)
GA_EXPERIMENT_GENS = 300


def _evolveMusicGenomeCustom(weights, generations):
    """Run the GA with a custom weight config. Temporarily swaps module-level
    weight globals so the existing fullCompFitness function picks them up,
    then restores the originals when done."""
    import sys
    mod = sys.modules[__name__]

    saved = {k: getattr(mod, k) for k in weights}
    for k, v in weights.items():
        setattr(mod, k, v)

    genome = musicGenome()
    genome.fitness = fullCompFitness(genome)
    history = [genome.fitness]

    try:
        for _ in range(generations):
            child = genome.copy()
            child.mutate()
            child.fitness = fullCompFitness(child)
            if child.fitness > genome.fitness:
                genome = child
            history.append(genome.fitness)
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)

    return genome, history


def runGAWeightExperiments(generations=GA_EXPERIMENT_GENS):
    print(f"\n=== Running GA Weight Experiments ({generations} gens each) ===")
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    results = {}
    for configName, weights in GA_EXPERIMENT_CONFIGS.items():
        print(f"  Running config: {configName} ...")
        genome, history = _evolveMusicGenomeCustom(weights, generations)
        results[configName] = (genome, history)
        print(f"    Final fitness: {genome.fitness:.1f}")

    # --- fitness curves ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for configName, (_, history) in results.items():
        ax.plot(history, label=configName)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('GA Weight Experiment — Fitness Over Generations')
    ax.legend(loc='lower right')
    plt.tight_layout()
    curvePath = os.path.join(GRAPHS_DIR, 'GA_Experiment_Curves.png')
    plt.savefig(curvePath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {curvePath}")

    # --- final fitness bar chart ---
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results.keys())
    finalFitness = [results[n][0].fitness for n in names]
    colors = plt.cm.tab10(range(len(names)))
    ax.bar(names, finalFitness, color=colors)
    ax.set_ylabel('Final Best Fitness')
    ax.set_title('GA Weight Experiment — Final Fitness Comparison')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right')
    plt.tight_layout()
    barPath = os.path.join(GRAPHS_DIR, 'GA_Experiment_Bar.png')
    plt.savefig(barPath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {barPath}")

    # --- voice plot + MIDI per config ---
    os.makedirs(MIDI_DIR, exist_ok=True)
    for configName, (genome, _) in results.items():
        safeName = configName.replace(' ', '_').replace('-', '_')
        voicePath = os.path.join(GRAPHS_DIR, f'GA_Experiment_{safeName}.png')
        plotVoices(genome, f'GA Experiment: {configName}', voicePath)

        expMelody = genomeToMus21(genome.melody, 'melody')
        expUpper  = genomeToMus21(genome.upper,  'upper')
        expMiddle = genomeToMus21(genome.middle, 'middle')
        expBass   = genomeToMus21(genome.bass,   'bass')
        expScore  = stream.Score()
        expScore.insert(0, instrument.Piano())
        expScore.append(expMelody)
        expScore.append(expUpper)
        expScore.append(expMiddle)
        expScore.append(expBass)
        midiPath = os.path.join(MIDI_DIR, f'GA_Experiment_{safeName}.mid')
        expScore.write('midi', midiPath)
        print(f"  Saved MIDI: {midiPath}")

    print("=== GA Weight Experiments complete ===\n")
    return results


# ***************************************************************
# ----------------------- MAIN LOOP ------------------------------
# ***************************************************************
# heres the main loop for the overall genome generation -> evoloution ->
# -> then to music21 and MIDI files

# run the GA evolution loop
evolvedComp = evolveMusicGenome()

# run the stage 1 pso work loop
chordsOptimizedComp = stage1PSOWork(evolvedComp)

# run the stage 2 pso work loop
finalComposition = stage2PSOWork(chordsOptimizedComp)

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
os.makedirs(MIDI_DIR, exist_ok=True)
melodyOnly.write('midi', os.path.join(MIDI_DIR, 'melody.mid'))

# one with just the upper and melody:
upperAndMelody = stream.Score()
upperAndMelody.append(melodyOnly)
upperAndMelody.append(upperOnly)
upperAndMelody.write('midi', os.path.join(MIDI_DIR, 'upper.mid'))

# one with just the middle and melody:
middleAndMelody = stream.Score()
middleAndMelody.append(melodyOnly)
middleAndMelody.append(middleOnly)
middleAndMelody.write('midi', os.path.join(MIDI_DIR, 'middle.mid'))

# one with just the bass and melody:
bassAndMelody = stream.Score()
bassAndMelody.append(melodyOnly)
bassAndMelody.append(bassOnly)
bassAndMelody.write('midi', os.path.join(MIDI_DIR, 'bass.mid'))

# and finally with all four together BEFORE THE PSO:
fullComp = stream.Score()
fullComp.insert(0, instrument.Piano())
fullComp.append(melodyOnly)
fullComp.append(upperOnly)
fullComp.append(middleOnly)
fullComp.append(bassOnly)
fullComp.write('midi', os.path.join(MIDI_DIR, 'GA_ONLY.mid'))

# Stage 1 PSO:::
# now for the PSO stage 1 for each line genome to music 21 format
S1melody = genomeToMus21(chordsOptimizedComp.melody, 'melody')
S1upper = genomeToMus21(chordsOptimizedComp.upper, 'upper')
S1middle = genomeToMus21(chordsOptimizedComp.middle, 'middle')
S1bass = genomeToMus21(chordsOptimizedComp.bass, 'bass')

# now for stage 1 music21 to midi
S1Full = stream.Score()
S1Full.insert(0, instrument.Piano())
S1Full.append(S1melody)
S1Full.append(S1upper)
S1Full.append(S1middle)
S1Full.append(S1bass)
S1Full.write('midi', os.path.join(MIDI_DIR, 'Stage1.mid'))

# Stage 2 PSO:::
# now for the PSO stage 2 for each line genome to music 21 format
S2melody = genomeToMus21(finalComposition.melody, 'melody')
S2upper = genomeToMus21(finalComposition.upper, 'upper')
S2middle = genomeToMus21(finalComposition.middle, 'middle')
S2bass = genomeToMus21(finalComposition.bass, 'bass')

# now for stage 2 music21 to midi
S2Full = stream.Score()  
S2Full.insert(0, instrument.Piano())
S2Full.append(S2melody)
S2Full.append(S2upper)
S2Full.append(S2middle)
S2Full.append(S2bass)
S2Full.write('midi', os.path.join(MIDI_DIR, 'FINAL_S1_and_S2.mid'))

# ***************************************************************
# ----------------------- SAVE VOICE PLOTS ------------------------------
# ***************************************************************
os.makedirs(GRAPHS_DIR, exist_ok=True)
plotVoices(evolvedComp,         'GA — Voice Lines',         os.path.join(GRAPHS_DIR, 'GA.png'))
plotVoices(chordsOptimizedComp, 'Stage1 PSO — Voice Lines', os.path.join(GRAPHS_DIR, 'Stage1.png'))
plotVoices(finalComposition,    'Final PSO — Voice Lines',  os.path.join(GRAPHS_DIR, 'Final_PSO.png'))

# ***************************************************************
# ----------------------- GA WEIGHT EXPERIMENT RUN ------------------------------
# ***************************************************************
runGAWeightExperiments()



    
