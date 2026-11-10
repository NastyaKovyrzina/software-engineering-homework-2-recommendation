from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", choices=[
        "all", "popular", "also-watched", "content", "item-knn", "two-tower"
    ])
    parser.add_argument("--stage", choices=["val", "test"], default="test")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--user", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs должно быть положительным")

    import numpy as np
    import pandas as pd
    from recsys.baselines import PopularRecommender, AlsoWatchedRecommender
    from recsys.classic import ContentBasedRecommender, ItemKNNRecommender
    from recsys.data import load_movielens
    from recsys.metrics import evaluate

    data = load_movielens()
    matches = data.users.index[data.users["user_id"] == args.user].to_numpy()
    if not len(matches):
        parser.error("Такого UserID нет в MovieLens")
    user = int(matches[0])
    history, target = data.stage(args.stage)
    print(f"MovieLens: {data.n_users} пользователей, {data.n_items} фильмов")
    print(f"Обучение: {len(history)}, {args.stage}: {len(target)} взаимодействий")

    models = {
        "popular": PopularRecommender,
        "also-watched": AlsoWatchedRecommender,
        "content": ContentBasedRecommender,
        "item-knn": ItemKNNRecommender,
        "two-tower": None,
    }
    selected = list(models) if args.model == "all" else [args.model]
    weights = ROOT / "artifacts" / "two_tower.pt"
    results = {}
    trained_two_tower = False

    for key in selected:
        fit = True
        if key == "two-tower":
            import torch
            from recsys.two_tower import TwoTowerRecommender
            if weights.exists() and not args.train and args.stage == "test":
                model = TwoTowerRecommender.from_pretrained(weights, data, history)
                fit = False
                print("TwoTower: загружены сохранённые веса")
            else:
                model = TwoTowerRecommender(epochs=args.epochs)
                print(f"TwoTower: обучение {args.epochs} эпох на {model.device}")
        else:
            model = models[key]()

        metrics = evaluate(model, data, stage=args.stage, fit=fit)
        results[model.name] = metrics
        print(f"{model.name}: NDCG@10={metrics['ndcg@10']:.4f}, "
              f"Recall@10={metrics['recall@10']:.4f}, "
              f"HitRate@10={metrics['hitrate@10']:.4f}, "
              f"Coverage@10={metrics['coverage@10']:.4f}")
        print(f"Рекомендации пользователю {args.user}:")
        for rank, item in enumerate(model.recommend(np.array([user]), 10)[0], 1):
            print(f"  {rank:2}. {data.items.loc[item, 'title']}")

        if key == "two-tower" and fit and args.stage == "test":
            weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.net.state_dict(), weights)
            trained_two_tower = True

    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    suffix = args.stage if args.model == "all" else f"{args.stage}_{args.model}"
    filename = output / f"metrics_{suffix}.csv"
    pd.DataFrame.from_dict(results, orient="index").rename_axis("model").to_csv(filename)
    (output / f"run_{suffix}.json").write_text(json.dumps({
        "stage": args.stage,
        "models": list(results),
        "epochs": args.epochs if trained_two_tower or (
            "two-tower" in selected and args.stage == "val"
        ) else None,
        "description": "Результаты вычислены main.py",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результаты сохранены: {filename}")


if __name__ == "__main__":
    main()
