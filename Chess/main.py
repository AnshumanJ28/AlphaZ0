"""
chesmain.py
Chess UI — pygame front-end with:
  - Main menu: Pass & Play  vs  AlphaZ0 Bot
  - Color selection when playing vs bot (White / Black)
  - MCTS bot integration (loads chess_net_best.pt if available)
  - Move log sidebar in algebraic notation (e4, Nf3, O-O …)
  - All original features: highlights, dots, promotion dialog, undo, restart
"""

import sys
import os
import threading
import pygame as p
import chesseng as ce

# ── optional bot imports (graceful fallback if model missing) ─────────────────
try:
    import torch
    from NeuralNet import ChessNet
    from mcts import MCTS
    BOT_AVAILABLE = True
except ImportError:
    BOT_AVAILABLE = False

# ── layout ────────────────────────────────────────────────────────────────────
BOARD_SIZE  = 512
SQ_SIZE     = BOARD_SIZE // 8
SIDEBAR_W   = 200          # move-log panel
STATUS_H    = 36
DIMENSION   = 8
MAX_FPS     = 30

# Full window: board + sidebar, status bar below board only
WIN_W   = BOARD_SIZE + SIDEBAR_W
WIN_H   = BOARD_SIZE + STATUS_H

# ── palette ───────────────────────────────────────────────────────────────────
LIGHT_SQ      = p.Color(240, 217, 181)
DARK_SQ       = p.Color(181, 136,  99)
BG_DARK       = p.Color( 22,  21,  18)   # menu / sidebar background
PANEL_BG      = p.Color( 32,  30,  27)   # sidebar
PANEL_BORDER  = p.Color( 55,  52,  46)
STATUS_BG     = p.Color( 28,  27,  24)
STATUS_FG     = p.Color(210, 200, 185)
CHECK_FG      = p.Color(220,  70,  60)
ACCENT        = p.Color(194, 145,  75)   # warm gold
ACCENT_DIM    = p.Color(120,  88,  42)
TEXT_MAIN     = p.Color(230, 220, 205)
TEXT_DIM      = p.Color(130, 120, 105)
WHITE_BTN     = p.Color(240, 235, 225)
BLACK_BTN     = p.Color( 45,  43,  38)
BOT_TAG_BG    = p.Color( 50,  38,  20)
BOT_TAG_FG    = p.Color(194, 145,  75)

IMAGES: dict = {}

# ── bot state (shared between threads) ───────────────────────────────────────
_bot_model   = None
_bot_mcts    = None
_bot_device  = None
_bot_loading = False
_bot_loaded  = False


def _load_bot_model():
    global _bot_model, _bot_mcts, _bot_device, _bot_loading, _bot_loaded
    _bot_loading = True
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = ChessNet().to(device)
        ckpt_paths = [
            "checkpoints/chess_net_best.pt",
            "chess_net_best.pt",
            "checkpoints/chess_net_final.pt",
        ]
        loaded = False
        for path in ckpt_paths:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                loaded = True
                break
        if not loaded:
            # Random weights — still playable, just very weak
            pass
        model.eval()
        _bot_model  = model
        _bot_device = device
        _bot_mcts   = MCTS(model, device, num_simulations=80)
        _bot_loaded = True
    except Exception as e:
        print(f"[Bot] Failed to load: {e}")
    finally:
        _bot_loading = False


def load_images() -> None:
    base   = os.path.dirname(os.path.abspath(__file__))
    pieces = ["wP","wR","wN","wB","wQ","wK","bP","bR","bN","bB","bQ","bK"]
    for piece in pieces:
        path = os.path.join(base, "Image", piece + ".png")
        raw  = p.image.load(path).convert_alpha()
        IMAGES[piece] = p.transform.smoothscale(raw, (SQ_SIZE, SQ_SIZE))


# ═══════════════════════════════════════════════════════════════════════════════
#  MENU SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

def menu_screen(screen: p.Surface) -> dict:
    """
    Blocking menu.  Returns a config dict:
      { 'mode': 'pvp' | 'bot',
        'player_color': 'w' | 'b'   (only meaningful for 'bot' mode) }
    """
    p.display.set_caption("AlphaZ0 Chess")
    clock = p.time.Clock()

    title_font   = p.font.SysFont("georgia",       52, bold=True)
    sub_font     = p.font.SysFont("segoeui",        14)
    btn_font     = p.font.SysFont("segoeui",        17, bold=True)
    tag_font     = p.font.SysFont("consolas",       11)
    label_font   = p.font.SysFont("segoeui",        13)
    small_font   = p.font.SysFont("segoeui",        11)

    W, H = screen.get_size()

    # ── button geometry ───────────────────────────────────────────────────────
    btn_w, btn_h = 260, 58
    gap          = 18
    total_h      = btn_h * 2 + gap
    by_pvp       = H // 2 - 10
    by_bot       = by_pvp + btn_h + gap
    bx           = (W - btn_w) // 2

    rect_pvp = p.Rect(bx, by_pvp, btn_w, btn_h)
    rect_bot = p.Rect(bx, by_bot, btn_w, btn_h)

    # Color-picker shown only after Bot is clicked
    show_color_picker = False
    color_choice      = "w"       # default

    cp_w, cp_h = 120, 46
    cp_gap      = 12
    cp_y        = by_bot + btn_h + 22
    rect_white  = p.Rect(bx,                cp_y, cp_w, cp_h)
    rect_black  = p.Rect(bx + cp_w + cp_gap, cp_y, cp_w, cp_h)
    rect_go     = p.Rect(bx + 30, cp_y + cp_h + 14, btn_w - 60, 42)

    hover_pvp = hover_bot = hover_white = hover_black = hover_go = False

    # Subtle board pattern for background
    tile = p.Surface((32, 32))

    def draw_bg():
        screen.fill(BG_DARK)
        for r in range(H // 32 + 1):
            for c in range(W // 32 + 1):
                if (r + c) % 2 == 0:
                    p.draw.rect(screen, p.Color(28, 27, 24), (c*32, r*32, 32, 32))

    def draw_btn(rect, text, hovered, accent=False, tag=None):
        bg  = ACCENT if accent else (p.Color(50, 48, 42) if hovered else p.Color(38, 36, 32))
        bdr = ACCENT if accent else (p.Color(80, 76, 68) if hovered else p.Color(55, 52, 46))
        p.draw.rect(screen, bg,  rect, border_radius=8)
        p.draw.rect(screen, bdr, rect, width=1, border_radius=8)
        tc  = BG_DARK if accent else TEXT_MAIN
        s   = btn_font.render(text, True, tc)
        screen.blit(s, s.get_rect(center=rect.center))
        if tag:
            ts  = tag_font.render(tag, True, BOT_TAG_FG)
            tr  = ts.get_rect(topright=(rect.right - 10, rect.top + 8))
            p.draw.rect(screen, BOT_TAG_BG, tr.inflate(10, 4), border_radius=3)
            screen.blit(ts, tr)

    running = True
    while running:
        mx, my = p.mouse.get_pos()
        hover_pvp   = rect_pvp.collidepoint(mx, my)
        hover_bot   = rect_bot.collidepoint(mx, my)
        hover_white = rect_white.collidepoint(mx, my) if show_color_picker else False
        hover_black = rect_black.collidepoint(mx, my) if show_color_picker else False
        hover_go    = rect_go.collidepoint(mx, my)    if show_color_picker else False

        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit(); sys.exit()

            if event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                if rect_pvp.collidepoint(mx, my):
                    return {"mode": "pvp", "player_color": "w"}

                if rect_bot.collidepoint(mx, my):
                    show_color_picker = True

                if show_color_picker:
                    if rect_white.collidepoint(mx, my):
                        color_choice = "w"
                    if rect_black.collidepoint(mx, my):
                        color_choice = "b"
                    if rect_go.collidepoint(mx, my):
                        return {"mode": "bot", "player_color": color_choice}

        # ── draw ─────────────────────────────────────────────────────────────
        draw_bg()

        # Title
        title = title_font.render("AlphaZ0", True, ACCENT)
        screen.blit(title, title.get_rect(centerx=W//2, centery=H//2 - 130))

        sub = sub_font.render("neural chess engine", True, TEXT_DIM)
        screen.blit(sub, sub.get_rect(centerx=W//2, centery=H//2 - 82))

        # Separator
        p.draw.line(screen, PANEL_BORDER,
                    (W//2 - 80, H//2 - 58), (W//2 + 80, H//2 - 58), 1)

        # Buttons
        draw_btn(rect_pvp, "Pass & Play", hover_pvp)
        draw_btn(rect_bot, "Play vs AlphaZ0", hover_bot, accent=not show_color_picker,
                 tag="AI")

        # Color picker
        if show_color_picker:
            prompt = label_font.render("Play as:", True, TEXT_DIM)
            screen.blit(prompt, (bx, cp_y - 22))

            # White option
            is_w = color_choice == "w"
            wb   = p.Color(255,255,255) if (hover_white or is_w) else p.Color(200,195,185)
            p.draw.rect(screen, wb, rect_white, border_radius=8)
            if is_w:
                p.draw.rect(screen, ACCENT, rect_white, width=2, border_radius=8)
            ws = btn_font.render("White", True, BG_DARK)
            screen.blit(ws, ws.get_rect(center=rect_white.center))

            # Black option
            is_b = color_choice == "b"
            bb   = p.Color(60,58,52) if (hover_black or is_b) else p.Color(45,43,38)
            p.draw.rect(screen, bb, rect_black, border_radius=8)
            if is_b:
                p.draw.rect(screen, ACCENT, rect_black, width=2, border_radius=8)
            bs = btn_font.render("Black", True, TEXT_MAIN)
            screen.blit(bs, bs.get_rect(center=rect_black.center))

            # Go button
            go_bg  = ACCENT if hover_go else ACCENT_DIM
            go_clr = BG_DARK if hover_go else TEXT_MAIN
            p.draw.rect(screen, go_bg, rect_go, border_radius=8)
            gs_txt = btn_font.render("Start Game  ›", True, go_clr)
            screen.blit(gs_txt, gs_txt.get_rect(center=rect_go.center))

        # Footer hint
        hint = small_font.render("v1.0  ·  AlphaZero-style MCTS + A3C", True,
                                 p.Color(60, 57, 50))
        screen.blit(hint, hint.get_rect(centerx=W//2, bottom=H - 14))

        p.display.flip()
        clock.tick(MAX_FPS)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN GAME LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p.init()
    screen = p.display.set_mode((WIN_W, WIN_H))
    p.display.set_caption("AlphaZ0 Chess")
    clock  = p.time.Clock()

    load_images()

    # Fonts
    status_font  = p.font.SysFont("segoeui",  13)
    label_font   = p.font.SysFont("segoeui",  11, bold=True)
    log_font     = p.font.SysFont("consolas", 12)
    log_hdr_font = p.font.SysFont("segoeui",  12, bold=True)
    thinking_font= p.font.SysFont("segoeui",  13)

    # Reusable alpha surfaces
    overlay  = p.Surface((SQ_SIZE, SQ_SIZE), p.SRCALPHA)
    dot_surf = p.Surface((SQ_SIZE, SQ_SIZE), p.SRCALPHA)
    dot_surf.fill((0,0,0,0))
    p.draw.circle(dot_surf, (20,20,20,90),
                  (SQ_SIZE//2, SQ_SIZE//2), SQ_SIZE//6)

    while True:   # outer loop — allows returning to menu after game ends
        # ── show menu ────────────────────────────────────────────────────────
        config = menu_screen(screen)
        mode         = config["mode"]           # 'pvp' | 'bot'
        player_color = config["player_color"]   # 'w' | 'b'

        # ── if bot mode, start loading model in background ────────────────
        global _bot_loaded, _bot_loading
        if mode == "bot" and BOT_AVAILABLE and not _bot_loaded:
            t = threading.Thread(target=_load_bot_model, daemon=True)
            t.start()

        # ── game state ───────────────────────────────────────────────────────
        gs           = ce.GameState()
        valid_moves  = gs.get_valid_moves()
        game_status  = "ongoing"
        sq_selected  = ()
        player_clicks= []
        last_move    = None
        move_made    = False
        san_log: list[str] = []   # flat list of SAN strings (white then black, alternating)
        log_scroll   = 0          # top row index in sidebar

        # Bot thinking state
        bot_thinking = False
        bot_move_result = [None]  # shared list for thread result

        def is_player_turn() -> bool:
            """True when the human should interact."""
            if mode == "pvp":
                return True
            # bot mode: player acts only on their own color's turn
            color_turn = "w" if gs.white_to_move else "b"
            return color_turn == player_color

        def trigger_bot_move():
            nonlocal bot_thinking
            if bot_thinking:
                return
            bot_thinking = True
            bot_move_result[0] = None

            def _think():
                if _bot_loaded and _bot_mcts is not None:
                    _, best = _bot_mcts.get_move_probs(gs, temperature=1e-3)
                    bot_move_result[0] = best
                else:
                    # Fallback: pick a random legal move
                    import random
                    moves = gs.get_valid_moves()
                    bot_move_result[0] = random.choice(moves) if moves else None

            threading.Thread(target=_think, daemon=True).start()

        # ── resize window to include sidebar ─────────────────────────────────
        screen = p.display.set_mode((WIN_W, WIN_H))
        p.display.set_caption("AlphaZ0 Chess")

        running = True
        while running:

            # ── bot move check ────────────────────────────────────────────────
            if mode == "bot" and not is_player_turn() and game_status == "ongoing":
                if not bot_thinking:
                    trigger_bot_move()
                elif bot_move_result[0] is not None:
                    best = bot_move_result[0]
                    bot_thinking = False
                    if best is not None:
                        # record SAN before making move
                        san = _get_san_simple(gs, best)
                        gs.make_move(best)
                        san_log.append(san)
                        last_move = best
                        move_made = True

            # ── events ───────────────────────────────────────────────────────
            for event in p.event.get():
                if event.type == p.QUIT:
                    p.quit(); sys.exit()

                elif event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if my >= BOARD_SIZE or mx >= BOARD_SIZE:
                        continue   # clicked sidebar or status bar
                    if not is_player_turn() or game_status != "ongoing":
                        continue

                    col = mx // SQ_SIZE
                    row = my // SQ_SIZE
                    if not (0 <= row < DIMENSION and 0 <= col < DIMENSION):
                        continue

                    # If board is flipped (player is black), convert coords
                    if mode == "bot" and player_color == "b":
                        row = 7 - row
                        col = 7 - col

                    if sq_selected == (row, col):
                        sq_selected   = ()
                        player_clicks = []
                        continue

                    piece = gs.board[row][col]
                    if not player_clicks:
                        if piece == "--":
                            continue
                        if gs.white_to_move and piece[0] != "w":
                            continue
                        if not gs.white_to_move and piece[0] != "b":
                            continue

                    sq_selected = (row, col)
                    player_clicks.append((row, col))

                    if len(player_clicks) == 2:
                        bare = ce.Move(player_clicks[0], player_clicks[1], gs.board)
                        validated = next((m for m in valid_moves if m == bare), None)

                        if validated is not None:
                            promo = "Q"
                            if (validated.piece_moved[1] == "P"
                                    and validated.end_row in (0, 7)):
                                promo = promotion_dialog(
                                    screen, status_font, validated.piece_moved[0])
                            san = _get_san_simple(gs, validated)
                            gs.make_move(validated, promotion_choice=promo)
                            san_log.append(san)
                            last_move = validated
                            move_made = True

                        sq_selected   = ()
                        player_clicks = []

                elif event.type == p.KEYDOWN:
                    if event.key == p.K_z and not bot_thinking:
                        # Undo: in bot mode, undo 2 moves (player + bot)
                        undo_count = 2 if (mode == "bot" and len(gs.move_log) >= 2) else 1
                        for _ in range(undo_count):
                            if gs.move_log:
                                gs.undo_move()
                                if san_log:
                                    san_log.pop()
                        last_move     = gs.move_log[-1] if gs.move_log else None
                        sq_selected   = ()
                        player_clicks = []
                        move_made     = True

                    elif event.key == p.K_r:
                        running = False   # go back to menu

                    elif event.key == p.K_ESCAPE:
                        running = False

                elif event.type == p.MOUSEWHEEL:
                    log_scroll = max(0, log_scroll - event.y)

            # ── recompute after board change ──────────────────────────────────
            if move_made:
                valid_moves = gs.get_valid_moves()
                game_status = gs.get_status(valid_moves)
                move_made   = False

            # ── destination dots ──────────────────────────────────────────────
            dest_squares: set = set()
            if sq_selected and is_player_turn():
                sr, sc = sq_selected
                for m in valid_moves:
                    if m.start_row == sr and m.start_col == sc:
                        dest_squares.add((m.end_row, m.end_col))

            # ── draw ─────────────────────────────────────────────────────────
            flipped = (mode == "bot" and player_color == "b")

            draw_board(screen, sq_selected, dest_squares, last_move,
                       overlay, dot_surf, flipped)
            draw_coord_labels(screen, label_font, flipped)
            draw_pieces(screen, gs.board, flipped)
            draw_sidebar(screen, san_log, log_font, log_hdr_font,
                         log_scroll, mode, player_color,
                         bot_thinking, _bot_loading, thinking_font)
            draw_status_bar(screen, gs, status_font, game_status, mode)

            clock.tick(MAX_FPS)
            p.display.flip()


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_board(screen, sq_selected, dest_squares, last_move,
               overlay, dot_surf, flipped):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            dr = 7 - r if flipped else r
            dc = 7 - c if flipped else c

            color = LIGHT_SQ if (r + c) % 2 == 0 else DARK_SQ
            rect  = p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            p.draw.rect(screen, color, rect)

            if last_move is not None:
                lsr = 7 - last_move.start_row if flipped else last_move.start_row
                lsc = 7 - last_move.start_col if flipped else last_move.start_col
                ler = 7 - last_move.end_row   if flipped else last_move.end_row
                lec = 7 - last_move.end_col   if flipped else last_move.end_col
                if (r, c) in ((lsr, lsc), (ler, lec)):
                    overlay.fill((50, 205, 50, 80))
                    screen.blit(overlay, rect)

            if sq_selected:
                ssr = 7 - sq_selected[0] if flipped else sq_selected[0]
                ssc = 7 - sq_selected[1] if flipped else sq_selected[1]
                if (r, c) == (ssr, ssc):
                    overlay.fill((255, 215, 0, 130))
                    screen.blit(overlay, rect)

            if (dr, dc) in dest_squares:
                screen.blit(dot_surf, rect)


def draw_coord_labels(screen, font, flipped):
    files = "abcdefgh"
    for i in range(DIMENSION):
        fi = 7 - i if flipped else i
        ri = 7 - i if flipped else i
        col  = DARK_SQ if i % 2 == 0 else LIGHT_SQ
        fs   = font.render(files[fi], True, col)
        screen.blit(fs, (i * SQ_SIZE + SQ_SIZE - 14, BOARD_SIZE - 16))
        rs   = font.render(str(ri + 1), True, col)
        screen.blit(rs, (2, (7 - i) * SQ_SIZE + 2))


def draw_pieces(screen, board, flipped):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            br = 7 - r if flipped else r
            bc = 7 - c if flipped else c
            piece = board[br][bc]
            if piece != "--":
                screen.blit(IMAGES[piece],
                            p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def draw_sidebar(screen, san_log, font, hdr_font, scroll,
                 mode, player_color, bot_thinking, bot_loading, thinking_font):
    # Sidebar background
    sidebar_rect = p.Rect(BOARD_SIZE, 0, SIDEBAR_W, WIN_H)
    p.draw.rect(screen, PANEL_BG, sidebar_rect)
    p.draw.line(screen, PANEL_BORDER,
                (BOARD_SIZE, 0), (BOARD_SIZE, WIN_H), 1)

    x0  = BOARD_SIZE + 12
    y0  = 10

    # Header
    hdr = hdr_font.render("Move Log", True, ACCENT)
    screen.blit(hdr, (x0, y0))
    p.draw.line(screen, PANEL_BORDER,
                (BOARD_SIZE + 8, y0 + 20), (BOARD_SIZE + SIDEBAR_W - 8, y0 + 20), 1)
    y0 += 28

    # Mode badge
    if mode == "bot":
        pc  = "White" if player_color == "w" else "Black"
        tag = hdr_font.render(f"vs AlphaZ0 · {pc}", True, BOT_TAG_FG)
        screen.blit(tag, (x0, y0))
        y0 += 18

    p.draw.line(screen, PANEL_BORDER,
                (BOARD_SIZE + 8, y0), (BOARD_SIZE + SIDEBAR_W - 8, y0), 1)
    y0 += 8

    # Bot thinking indicator
    if bot_loading:
        dots = "." * ((p.time.get_ticks() // 400) % 4)
        ts = thinking_font.render(f"Loading model{dots}", True, TEXT_DIM)
        screen.blit(ts, (x0, y0))
        y0 += 20
    elif bot_thinking:
        dots = "." * ((p.time.get_ticks() // 300) % 4)
        ts = thinking_font.render(f"Thinking{dots}", True, ACCENT)
        screen.blit(ts, (x0, y0))
        y0 += 20

    # Clip region for move list
    clip_top    = y0
    clip_bottom = WIN_H - STATUS_H - 30
    clip_h      = clip_bottom - clip_top
    if clip_h <= 0:
        return

    # Build paired rows: [(num, white_san, black_san), ...]
    rows = []
    for i in range(0, len(san_log), 2):
        w = san_log[i]
        b = san_log[i+1] if i+1 < len(san_log) else ""
        rows.append((i//2 + 1, w, b))

    row_h   = 18
    visible = clip_h // row_h
    max_scroll = max(0, len(rows) - visible + 1)
    scroll  = min(scroll, max_scroll)

    screen.set_clip(p.Rect(BOARD_SIZE + 1, clip_top, SIDEBAR_W - 2, clip_h))

    for idx, (num, ws, bs) in enumerate(rows[scroll:scroll + visible + 1]):
        y = clip_top + idx * row_h

        # Alternating row tint
        if idx % 2 == 0:
            p.draw.rect(screen, p.Color(36, 34, 30),
                        p.Rect(BOARD_SIZE + 1, y, SIDEBAR_W - 2, row_h))

        # Row number
        ns = font.render(f"{num}.", True, TEXT_DIM)
        screen.blit(ns, (x0, y + 2))

        # White move
        wc = TEXT_MAIN if ws else TEXT_DIM
        ws_surf = font.render(ws, True, wc)
        screen.blit(ws_surf, (x0 + 32, y + 2))

        # Black move
        if bs:
            bs_surf = font.render(bs, True, TEXT_DIM)
            screen.blit(bs_surf, (x0 + 100, y + 2))

    screen.set_clip(None)

    # Scroll hint
    if len(rows) > visible:
        hint = font.render("scroll ↕", True, p.Color(55,52,46))
        screen.blit(hint, hint.get_rect(
            centerx=BOARD_SIZE + SIDEBAR_W // 2,
            bottom=WIN_H - STATUS_H - 4))

    # Keys hint
    hint2_font = p.font.SysFont("segoeui", 10)
    h2 = hint2_font.render("Z=undo   R=menu", True, p.Color(55, 52, 46))
    screen.blit(h2, h2.get_rect(
        centerx=BOARD_SIZE + SIDEBAR_W // 2,
        bottom=WIN_H - 4))


def draw_status_bar(screen, gs, font, game_status, mode):
    p.draw.rect(screen, STATUS_BG, p.Rect(0, BOARD_SIZE, BOARD_SIZE, STATUS_H))

    if game_status == "checkmate":
        winner = "Black" if gs.white_to_move else "White"
        text  = f"Checkmate — {winner} wins!    Z=undo  R=menu"
        color = CHECK_FG
    elif game_status == "stalemate":
        text  = "Stalemate — draw.    Z=undo  R=menu"
        color = STATUS_FG
    elif gs.in_check():
        side  = "White" if gs.white_to_move else "Black"
        text  = f"{side} is in check!"
        color = CHECK_FG
    else:
        side  = "White" if gs.white_to_move else "Black"
        text  = f"{side} to move"
        color = STATUS_FG

    surf = font.render(text, True, color)
    screen.blit(surf, (10, BOARD_SIZE + (STATUS_H - surf.get_height()) // 2))


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMOTION DIALOG
# ═══════════════════════════════════════════════════════════════════════════════

def promotion_dialog(screen, font, color: str) -> str:
    options = [("Q","Queen"),("R","Rook"),("B","Bishop"),("N","Knight")]
    btn_w, btn_h = 110, 52
    gap     = 10
    total_w = len(options) * btn_w + (len(options)-1) * gap
    ox      = (BOARD_SIZE - total_w) // 2
    oy      = (BOARD_SIZE - btn_h) // 2

    dim = p.Surface((BOARD_SIZE, BOARD_SIZE), p.SRCALPHA)
    dim.fill((0,0,0,160))
    screen.blit(dim, (0,0))

    rects = {}
    for i, (code, label) in enumerate(options):
        x    = ox + i*(btn_w+gap)
        rect = p.Rect(x, oy, btn_w, btn_h)
        rects[code] = rect
        p.draw.rect(screen, p.Color(45,43,38), rect, border_radius=8)
        p.draw.rect(screen, ACCENT,            rect, width=1, border_radius=8)
        key = color + code
        if key in IMAGES:
            img = p.transform.smoothscale(IMAGES[key], (34,34))
            screen.blit(img, img.get_rect(centerx=rect.centerx, top=rect.top+4))
        lbl = font.render(label, True, TEXT_MAIN)
        screen.blit(lbl, lbl.get_rect(centerx=rect.centerx, bottom=rect.bottom-4))

    p.display.flip()

    while True:
        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit(); sys.exit()
            if event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                for code, rect in rects.items():
                    if rect.collidepoint(event.pos):
                        return code
        p.time.Clock().tick(30)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMPLE SAN BUILDER (no temporary make/undo needed for log display)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_san_simple(gs: ce.GameState, move: ce.Move) -> str:
    """
    Lightweight SAN — accurate for 99% of cases.
    Castling, pawn captures, piece symbol + disambiguation.
    Check/mate suffix omitted for speed (avoids double move-gen).
    """
    if move.is_castle:
        return "O-O" if move.end_col == 6 else "O-O-O"

    files = "abcdefgh"
    ranks = "87654321"
    pt    = move.piece_moved[1]
    cap   = move.piece_captured != "--" or move.is_en_passant
    dest  = files[move.end_col] + ranks[move.end_row]

    if pt == "P":
        san = (files[move.start_col] + "x" if cap else "") + dest
        if move.end_row in (0, 7):
            san += "=Q"
        return san

    # Disambiguation
    ambig = [
        m for m in gs.get_valid_moves()
        if m != move
        and gs.board[m.start_row][m.start_col] == move.piece_moved
        and m.end_row == move.end_row and m.end_col == move.end_col
    ]
    dis = ""
    if ambig:
        same_file = any(m.start_col == move.start_col for m in ambig)
        same_rank = any(m.start_row == move.start_row for m in ambig)
        if same_file and same_rank:
            dis = files[move.start_col] + ranks[move.start_row]
        elif same_file:
            dis = ranks[move.start_row]
        else:
            dis = files[move.start_col]

    return pt + dis + ("x" if cap else "") + dest


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()