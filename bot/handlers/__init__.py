"""Сборка всех роутеров handlers."""

from bot.handlers import admin, commands, feedback, fsm, media, text


def get_routers():
    return [
        admin.router,
        feedback.router,
        commands.router,
        fsm.router,
        media.router,
        text.router,
    ]
