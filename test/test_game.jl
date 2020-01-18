function generate_boards(::Type{Game}, n) where Game
  player = RandomPlayer{Game}()
  rec = AlphaZero.Recorder{Game}()
  for i in 1:n
    self_play!(player, rec)
  end
  return Set(rec.boards)
end

function test_symmetries(::Type{Game}) where Game
  boards = generate_boards(Game, 100)
  for b in boards
    syms = GI.symmetries(Game, b)
    for sym in syms
      @test GI.test_symmetry(Game, b, sym)
    end
  end
end
