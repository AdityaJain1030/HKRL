namespace HKRL.Game
{
	public class SaveFileProxy
	{
		public static void LoadCompletedSave()
		{
			
		}

		//Add delagate to event to disable game saving when no savefile is loaded
		public static void DisableSaveGame(On.GameManager.orig_SaveGame orig, GameManager self)
		{
		}
	}
	
}