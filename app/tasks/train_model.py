from app.services.market_service import MarketService
from app.services.training import train_local_model


def main() -> None:
    service = MarketService()
    histories, names = service.get_training_inputs(limit=220)
    result = train_local_model(
        histories,
        names,
        provider_name=service.active_data_provider,
        configured_provider=service.settings.data_provider,
    )
    print(result["message"])
    print(
        f"train_rows={result['train_rows']} validation_rows={result['validation_rows']} "
        f"validation_ic={result['validation_ic']:.4f} "
        f"directional_accuracy={result['validation_directional_accuracy']:.4f}"
    )
    for item in result.get("comparison", []):
        print(
            f"model={item['model_name']} champion={item['is_champion']} "
            f"validation_ic={item['validation_ic']:.4f} "
            f"walk_forward_mean_ic={item['walk_forward_mean_ic']:.4f}"
        )
    for row in result["top_predictions"]:
        print(
            f"{row['rank']:>2} {row['symbol']} {row['name']} "
            f"pred_return_5d={row['predicted_return_5d']:.4f} score={row['score']:.3f}"
        )


if __name__ == "__main__":
    main()
