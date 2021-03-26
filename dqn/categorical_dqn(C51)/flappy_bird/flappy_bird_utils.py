import pygame


def load():
    # path of player with different states
    PLAYER_PATH = (
        'flappy_bird/images/redbird-upflap.png',
        'flappy_bird/images/redbird-midflap.png',
        'flappy_bird/images/redbird-downflap.png'
    )

    # path of background
    BACKGROUND_PATH = 'flappy_bird/images/background-black.png'

    # path of pipe
    PIPE_PATH = 'flappy_bird/images/pipe-green.png'

    IMAGES, HITMASKS = {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('flappy_bird/images/0.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/1.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/2.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/3.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/4.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/5.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/6.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/7.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/8.png').convert_alpha(),
        pygame.image.load('flappy_bird/images/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('flappy_bird/images/base.png').convert_alpha()

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, HITMASKS


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask
