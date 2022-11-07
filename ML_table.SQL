-- join required for project --
-- join team info into game_team_stats --
-- faceOffWinPercentage for both home and away are coming up with the same number for every entry on both sides. not sure why --

SELECT h.game_id, h.team_id_home, t.abbreviation AS "abbreviation_home", h.won_home, h."faceOffWinPercentage_home", h."powerPlayPrct_home", h."shootingPrct_home", h."penaltyKillPrct_home", h."savePrct_home"
INTO advanced_stats_home
FROM advanced_stats as h
JOIN team_info as t on h.team_id_home = t.team_id;

SELECT a.game_id, a.team_id_away, t.abbreviation AS "abbreviation_away", a.won_away, a."faceOffWinPercentage_away", a."powerPlayPrct_away", a."shootingPrct_away", a."penaltyKillPrct_away", a."savePrct_away"
INTO advanced_stats_away
FROM advanced_stats as a
JOIN team_info as t on a.team_id_away = t.team_id;

SELECT h.game_id, h.team_id_home, h."abbreviation_home", h.won_home, h."faceOffWinPercentage_home", h."powerPlayPrct_home", h."shootingPrct_home", h."penaltyKillPrct_home", h."savePrct_home", a.team_id_away, a."abbreviation_away", a.won_away, a."faceOffWinPercentage_away", a."powerPlayPrct_away", a."shootingPrct_away", a."penaltyKillPrct_away", a."savePrct_away"
INTO ml_table
FROM advanced_stats_home as h
JOIN advanced_stats_away as a on h.game_id = a.game_id