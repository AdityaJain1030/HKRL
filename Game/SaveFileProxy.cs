
using System;
using System.IO;
using System.Reflection;
using Newtonsoft.Json;
using UnityEngine;

namespace HKRL.Game
{
	public class SaveFileProxy
	{
		/// <summary>
		/// Load a completed save by deserializing a completed save JSON file and overwriting player data
		/// and scene data. Taken from HK Tag Mod.
		/// </summary>
		public static void LoadCompletedSave()
		{
			var saveResStream = Assembly.GetExecutingAssembly()
				.GetManifestResourceStream("HKRL.Resource.save_file.json");
			if (saveResStream == null)
			{
				HKRL.Instance.Log("Resource stream for save file is null");
				return;
			}

			var saveFileString = new StreamReader(saveResStream).ReadToEnd();

			// Deserialize the JSON file to a SaveGameData instance
			SaveGameData completedSaveGameData;
			try
			{
				completedSaveGameData = JsonConvert.DeserializeObject<SaveGameData>(saveFileString);
			}
			catch (Exception e)
			{
				HKRL.Instance.Log($"Could not deserialize completed save file, {e.GetType()}, {e.Message}");
				return;
			}

			// Overwrite the player data and scene data instances
			var gameManager = GameManager.instance;
			gameManager.playerData = PlayerData.instance = completedSaveGameData?.playerData;
			gameManager.sceneData = SceneData.instance = completedSaveGameData?.sceneData;
		}

		//Add delagate to event to disable game saving when no savefile is loaded
		public static void DisableSaveGame(On.GameManager.orig_SaveGame orig, GameManager self)
		{
		}
	}


}