using System.Collections;
using UnityEngine;

namespace HKRL.Game
{
	public static class SceneHooks
	{
		public static void StopAllTransitions()
		{

		}
		/// <summary>
		///  Loads a boss from the Hall of gods given the scene name
		/// </summary>
		/// <param name="scene_name">The name of the scene to load</param>
		public static IEnumerator LoadBossScene(string scene_name)
		{
			var HC = HeroController.instance;
			var GM = GameManager.instance;

			// On.BossSceneController.Start += (orig, self) => orig(self);

			//Copy paste of the FSM that loads a boss from HoG
			PlayMakerFSM.BroadcastEvent("DREAM ENTER");
			PlayerData.instance.dreamReturnScene = "GG_Workshop";
			PlayMakerFSM.BroadcastEvent("BOX DOWN DREAM");
			PlayMakerFSM.BroadcastEvent("CONVO CANCEL");
			PlayMakerFSM.BroadcastEvent("GG TRANSITION OUT");
			BossSceneController.SetupEvent = (self) =>
			{
				StaticVariableList.SetValue("bossSceneToLoad", scene_name);
				self.BossLevel = 1;
				self.DreamReturnEvent = "DREAM RETURN";
				self.OnBossSceneComplete += () => self.DoDreamReturn();
			};

			HC.ClearMPSendEvents();
			GM.TimePasses();
			GM.ResetSemiPersistentItems();
			HC.enterWithoutInput = true;
			HC.AcceptInput();

			GM.BeginSceneTransition(new GameManager.SceneLoadInfo
			{
				SceneName = scene_name,
				EntryGateName = "door_dreamEnter",
				EntryDelay = 0,
				Visualization = GameManager.SceneLoadVisualizations.GodsAndGlory,
				PreventCameraFadeOut = true
			});
			yield return FixSoul();
			yield return new WaitForSeconds(2f);
		}
		// quick soul hotfix 
		private static IEnumerator FixSoul()
		{
			yield return new WaitForFinishedEnteringScene();
			yield return null;
			yield return new WaitForSeconds(1f); //this line differenciates this function from ApplySettings
			HeroController.instance.AddMPCharge(1);
			HeroController.instance.AddMPCharge(-1);
		}

		public static bool ResetBossHealthAfterThreshold(On.HealthManager.orig_TakeDamage orig, HealthManager self, HitInstance hitInstance, int threshold, int resetTo)
		{
            bool wouldDie = false;
            if (self.hp - hitInstance.DamageDealt <= 50)
            {
                self.hp = resetTo;
                wouldDie = true;
            }
            orig(self, hitInstance);
            return wouldDie;
		}
	}
}